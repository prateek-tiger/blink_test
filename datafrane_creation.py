import pandas as pd
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from decimal import Decimal
from config import CONFIG
import datetime
import logging
import traceback
import numpy as np
from dateutil.relativedelta import relativedelta
from data_connection import ConnectionPool

class Filter(BaseModel):
    year: int = datetime.datetime.now().year
    month: int = datetime.datetime.now().month
    region: str = 'ALL'
    retailer: str = 'ALL'
    category: str = 'SALTY SNACKS'
    sub_category: str = 'ALL'
    product_universe: str = 'ALL'
    manufacturer: str = 'ALL'
    brand: str = 'ALL'
    comparison: str = "MTD"
    insight_type: str = "Hierarchical"
    kpi: str = "Revenue"

connect_pool = ConnectionPool()

class DataframeCreator:
    def __init__(self, filter:Filter):
        self.filter = filter
        self.current_data = None
        self.previous_data = None
        self.data_dict = {
            "country": {
                "overall": []
            },
            "region": {
                "overall": {},
                "top_performers": [],
                "bottom_performers": []
            },
            "sub_category": {
                "overall": {},
                "top_performers": [],
                "bottom_performers": []
            },
            "brand": {
                "overall": {},
                "top_performers": [],
                "bottom_performers": []
            },"sub_brand": {
                "overall": {},
                "top_performers": [],
                "bottom_performers": []
            },
            "flavour": {
                "overall": {},
                "top_performers": [],
                "bottom_performers": []
            },
            "pack_size": {
                "overall": {},
                "top_performers": [],
                "bottom_performers": []
            },
            "pack_group": {
                "overall": {},
                "top_performers": [],
                "bottom_performers": []
            }
        }
        self.merged_data = {

        }
        self.current_start_month = 0
        self.current_start_year = 0
        self.current_end_month = 0
        self.current_end_year = 0

        self.previous_start_month = 0
        self.previous_start_year = 0
        self.previous_end_month = 0
        self.previous_end_year = 0

    def prepare_comparison_time(self):

        if self.filter.comparison == 'MTD': # MTD vs PAGO

            # CURRENT DATA
            self.current_start_month = self.filter.month
            self.current_start_year = self.filter.year
            end_date = datetime.date(self.current_start_year, self.current_start_month, 1) + relativedelta(months=1)
            self.current_end_month = end_date.month
            self.current_end_year = end_date.year

            # PREVIOUS DATA
            start_date = datetime.date(self.current_start_year, self.current_start_month, 1) - relativedelta(months=1)
            self.previous_start_month = start_date.month
            self.previous_start_year = start_date.year
            self.previous_end_month = self.filter.month
            self.previous_end_year = self.filter.year

        elif self.filter.comparison == 'YTD': # YTD vs YAGO
            # CURRENT DATA
            self.current_start_month = 1
            self.current_start_year = self.filter.year
            end_date = datetime.date(self.filter.year, self.filter.month, 1) + relativedelta(months=1)
            self.current_end_month = end_date.month
            self.current_end_year = end_date.year

            # PREVIOUS DATA
            self.previous_start_month = 1
            self.previous_start_year = self.filter.year - 1
            # end_date = datetime.date(self.previous_start_year, self.previous_start_month, 1) + relativedelta(months=12)
            self.previous_end_year = self.current_end_year-1
            self.previous_end_month = self.current_end_month

        elif self.filter.comparison == 'MAT': # MAT vs MAT LY
            # CURRENT DATA
            start_date = datetime.date(self.filter.year, self.filter.month, 1) - relativedelta(months=11)
            self.current_start_month = start_date.month
            self.current_start_year = start_date.year
            end_date = datetime.date(self.filter.year, self.filter.month, 1) + relativedelta(months=1)
            self.current_end_month = end_date.month
            self.current_end_year = end_date.year

            # PREVIOUS DATA
            self.previous_start_month = self.current_start_month
            self.previous_start_year = self.current_start_year - 1
            self.previous_end_year = self.current_end_year - 1
            self.previous_end_month = self.current_end_month
        else:
            pass

        return {
            "current_start_month": self.current_start_month,
            "current_start_year": self.current_start_year,
            "current_end_month": self.current_end_month,
            "current_end_year": self.current_end_year,
            "previous_start_month": self.previous_start_month,
            "previous_start_year":  self.previous_start_year,
            "previous_end_month": self.previous_end_month,
            "previous_end_year": self.previous_end_year
        }

    def convert_decimal_columns_to_float(self, df):
        """
        Finds all columns in a DataFrame that contain Decimal objects
        and converts them to a float64 dtype.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with Decimal columns converted to float.
        """
        # Create a copy to avoid modifying the original DataFrame in place
        df_converted = df.copy()

        for col in df_converted.columns:
            # Check if the column's dtype is 'object'. Decimal columns are stored this way.
            if df_converted[col].dtype == 'object':

                # Safely check the type of the first non-null value in the column
                non_null_values = df_converted[col].dropna()
                if not non_null_values.empty and isinstance(non_null_values.iloc[0], Decimal):

                    # If it's a Decimal, convert the entire column to float
                    df_converted[col] = df_converted[col].astype(float)

        return df_converted

    def get_updated_columns_and_groupby(self, category: str) -> tuple:
        """
        Modifies the SELECT columns and GROUP BY clause based on applied filters.
        Injects filter-specific columns and group-by parts from CONFIG['filter_sql'],
        and removes duplicates from both lists.

        Parameters:
        - category (str): The category key in CONFIG['sql'], e.g., "region", "city", etc.

        Returns:
        - tuple: (columns_to_use_str, group_by_clause_str)
        """

        def deduplicate_preserve_order(seq):
            seen = set()
            return [x for x in seq if not (x in seen or seen.add(x))]

        base_columns = CONFIG['sql'][category]['columns_to_use']
        group_by_clause = CONFIG['sql'][category]['group_by_clause'].strip()

        columns_to_use = base_columns.copy()
        inserted_group_by_parts = []

        # Only process these filters
        filters_to_check = ['retailer', 'sub_category', 'brand', 'manufacturer']

        for filter_key in filters_to_check:
            filter_value = getattr(self.filter, filter_key, "ALL")
            if filter_value != "ALL":
                filter_config = CONFIG['filter_sql'].get(filter_key, {})
                filter_columns = filter_config.get('columns_to_use', [])
                filter_group_by = filter_config.get('group_by_clause', [])

                columns_to_use = filter_columns + columns_to_use
                inserted_group_by_parts.extend(filter_group_by)

        # Deduplicate both lists
        columns_to_use = deduplicate_preserve_order(columns_to_use)
        inserted_group_by_parts = deduplicate_preserve_order(inserted_group_by_parts)

        # Final SELECT
        select_str = ', '.join(columns_to_use)

        # Final GROUP BY clause
        if group_by_clause.upper().startswith("GROUP BY") and inserted_group_by_parts:
            inserted_group_by_str = ' '.join(inserted_group_by_parts).strip()
            group_by_clause = group_by_clause.replace(
                "GROUP BY", f"GROUP BY {inserted_group_by_str} ", 1
            ).rstrip(', ')

        return select_str, group_by_clause

    def fetch_data(self, category, current=True):
        try:
            columns_to_use, group_by_clause = self.get_updated_columns_and_groupby(category)
            connection = connect_pool.get_connection()
            cursor = connection.cursor()
            start_date = datetime.date(self.current_start_year, self.current_start_month, 1) if current else datetime.date(self.previous_start_year, self.previous_start_month, 1)
            end_date = datetime.date(self.current_end_year, self.current_end_month, 1) if current else datetime.date(self.previous_end_year, self.previous_end_month, 1)
            sql_query = f"SELECT {columns_to_use} {CONFIG['sql'][category]['from_clause']} {CONFIG['sql'][category]['join_clause']} {CONFIG['sql'][category]['where_clause']} {group_by_clause}"
            sql_query = sql_query.format(
                base_epos_sql = CONFIG['sql']['base_epos_sql']
            ).format(
                start_date = start_date,
                end_date = end_date
            )
            logging.info(sql_query)
            cursor.execute(sql_query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            df = self.convert_decimal_columns_to_float(df)
            return df
        except Exception as e:
            logging.error(f"Error: {e}")
            traceback.format_exc()
        finally:
            if cursor:
                cursor.close()
            if connection:
                connect_pool.release_connection(connection)

    def merge_data(self):
        """
        Merges current and previous period KPI data across different product hierarchy levels.

        For each category in the predefined list (e.g., brand, region, etc.), this method:
        1. Fetches the current and previous data.
        2. Merges both datasets on shared columns excluding metric fields.
        3. Calculates:
            - Revenue growth percentage.
            - Change in revenue share percentage points.
            - Absolute revenue change.
        4. Stores the merged and computed DataFrame in `self.merged_data`.

        Assumes `fetch_data(category, current)` method is implemented to retrieve data for each category.
        """

        # Initialize dictionaries to store data
        self.current_data = {}
        self.previous_data = {}

        # Define the levels of data to merge
        category_list = [
            "category", "region", "city", "sub_category", "brand",
            "sub_brand", "flavour", "pack_size", "pack_group"
        ]

        for key in category_list:
            # Fetch data for both current and previous periods
            self.current_data[key] = self.fetch_data(category=key, current=True)
            self.previous_data[key] = self.fetch_data(category=key, current=False)

            # Identify common columns to join on, excluding metric columns
            columns_to_merge_on = [
                col for col in self.current_data[key].columns
                if col not in ['revenue_k_aed', 'revenue_share_pct']
            ]

            # Merge current and previous datasets
            merged_df = self.current_data[key].merge(
                self.previous_data[key],
                on=columns_to_merge_on,
                suffixes=('', '_prev'),
                how='left'
            )

            # Replace missing values with 0
            merged_df.fillna(0, inplace=True)

            # Safely compute revenue growth only where previous revenue isn't zero
            safe_to_divide_mask = merged_df['revenue_k_aed_prev'] != 0
            merged_df['revenue_growth_pct'] = 100.0  # Default value if previous is 0
            merged_df.loc[safe_to_divide_mask, 'revenue_growth_pct'] = (
                (merged_df.loc[safe_to_divide_mask, 'revenue_k_aed'] -
                merged_df.loc[safe_to_divide_mask, 'revenue_k_aed_prev']) /
                merged_df.loc[safe_to_divide_mask, 'revenue_k_aed_prev']
            ) * 100

            # Calculate change in share percentage and absolute revenue
            merged_df['revenue_share_chg_pts'] = (
                merged_df['revenue_share_pct'] - merged_df['revenue_share_pct_prev']
            )
            merged_df['revenue_chg'] = (
                merged_df['revenue_k_aed'] - merged_df['revenue_k_aed_prev']
            )

            # Round all numerical values to 1 decimal place for clarity
            merged_df = merged_df.round(1)

            # Store the result
            self.merged_data[key] = merged_df

    def get_filter_condition(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns a boolean mask to filter the DataFrame based on selected filters.
        Applies only if values for brand, sub_category, manufacturer, or retailer are not "ALL".
        All conditions are combined using AND (&).

        Parameters:
        - df (pd.DataFrame): The DataFrame to apply the condition on.

        Returns:
        - pd.Series: Boolean mask to filter rows.
        """

        # Start with a default True condition
        condition = pd.Series([True] * len(df), index=df.index)

        # Filters to include
        filters_to_check = {
            'retailer': 'retailer',
            'sub_category': 'sub_category',
            'brand': 'brand',
            'manufacturer': 'manufacturer'  # both 'brand' and 'manufacturer' map to same column
        }

        for attr, col in filters_to_check.items():
            value = getattr(self.filter, attr, "ALL")
            if value != "ALL" and col in df.columns:
                condition &= (df[col] == value)

        return condition

    def get_region_market_leader(self) -> dict:
        """
        Groups the DataFrame by 'region', sums 'revenue_k_aed',
        and returns the region with the highest total revenue.

        Parameters:
        - df (pd.DataFrame): Input DataFrame with 'region' and 'revenue_k_aed' columns.

        Returns:
        - dict: A dictionary with 'region' and 'total_revenue_k_aed' keys.
        """
        if 'region' not in self.merged_data['region'].columns or 'revenue_k_aed' not in self.merged_data['region'].columns:
            raise ValueError("DataFrame must contain 'region' and 'revenue_k_aed' columns.")

        grouped_df = self.merged_data['region'].groupby('region', as_index=False)['revenue_k_aed'].sum()
        top_row = grouped_df.loc[grouped_df['revenue_k_aed'].idxmax()]

        return {
            "region": top_row['region'],
            "total_revenue_k_aed": top_row['revenue_k_aed']
        }

    def performance_ranking(self):
        """
        Computes top and bottom performers for various hierarchy levels based on revenue and revenue growth.
        Stores data in `self.performer_data` with:
        - overall: Best performer by revenue or revenue growth
        - top_performers: Top N by revenue change (level-specific)
        - bottom_performers: Bottom N by revenue change (level-specific)
        Filtering is based on a cascading hierarchy.
        """

        self.performer_data = {}

        # Define filter hierarchy for each level
        filter_hierarchy = {
            "category": [],
            "region": ['category'],
            "city": ['category', 'region'],
            "sub_category": ['category', 'region', 'city'],
            "brand": ['category', 'region', 'city', 'sub_category'],
            "sub_brand": ['category', 'region', 'city', 'sub_category'],
            "flavour": ['category', 'region', 'city', 'sub_category'],
            "pack_size": ['category', 'region', 'city', 'sub_category'],
            "pack_group": ['category', 'region', 'city', 'sub_category']
        }

        # Custom top_n / bottom_n settings per level
        top_bottom_n = {
            "region": (2, 1),
            "sub_category": (2, 2),
            "brand": (5, 5),
            "sub_brand": (5, 5),
            "flavour": (5, 5),
            "pack_size": (2, 1),
            "pack_group": (2, 2)
        }

        # Default fallback
        default_top_n = 2
        default_bottom_n = 3

        best_filters = {}

        # Utility to evaluate performance with custom top/bottom
        def evaluate_performance(df, key_col, top_n, bottom_n):
            if key_col == 'flavour':
                df = df[df['flavour'] != 'UNSPECIFIED']
            sorted_df = df.sort_values("revenue_chg", ascending=False)
            overall_best = sorted_df.iloc[0][key_col]
            top = sorted_df[key_col].head(top_n).tolist()
            bottom = sorted_df[key_col].tail(bottom_n).tolist()
            return {
                "overall": {"revenue_growth": overall_best},
                "top_performers": top,
                "bottom_performers": bottom
            }

        for level in filter_hierarchy:
            if level not in self.merged_data:
                continue

            df = self.merged_data[level][self.get_filter_condition(self.merged_data[level])].copy()

            # Apply parent-level filters
            for parent in filter_hierarchy[level]:
                if parent in best_filters:
                    df = df[df[parent] == best_filters[parent]]

            if df.empty or level not in df.columns:
                continue

            level_data = {}

            # Special logic for region
            if level == "region":
                df = df[df['region'] != 'NON-STORE']

                # top_rev = df.sort_values("revenue_k_aed", ascending=False).iloc[0]
                top_growth = df.sort_values("revenue_chg", ascending=False).iloc[0]
                market_leader_info = self.get_region_market_leader()
                level_data["overall"] = {
                    "revenue": market_leader_info['region'],
                    "revenue_growth": top_growth["region"] if self.filter.region == "ALL" else self.filter.region
                }

                best_filters["region"] = top_growth["region"] if self.filter.region == "ALL" else self.filter.region

                top_n, bottom_n = top_bottom_n.get(level, (default_top_n, default_bottom_n))
                sorted_df = df.sort_values("revenue_chg", ascending=False)
                level_data["top_performers"] = sorted_df["region"].head(top_n).tolist()
                level_data["bottom_performers"] = sorted_df["region"].tail(bottom_n).tolist()

            elif level in ["city", "sub_category", "brand", "sub_brand", "flavour", "pack_size", "pack_group"]:
                key_col = level
                top_n, bottom_n = top_bottom_n.get(level, (default_top_n, default_bottom_n))
                perf_data = evaluate_performance(df, key_col, top_n, bottom_n)

                best_filters[level] = perf_data["overall"]["revenue_growth"]
                level_data.update(perf_data)

            self.performer_data[level] = level_data

        logging.info(self.performer_data)

    def prepare_country_data(self):
        """
        Prepares and stores overall country-level data based on the 'category' merged dataset.
        Stores the data as a list of dictionaries under:
            self.data_dict['country']['overall']
        """
        if self.filter.region == "ALL":
            if 'category' not in self.merged_data:
                raise KeyError("'category' not found in merged_data. Ensure merge_data() is run before this.")

            self.data_dict['country']['overall'] = self.merged_data['category'][self.get_filter_condition(self.merged_data['category'])].to_dict(orient="records")
            logging.info("Data preparation completed for country")

    def prepare_region_city_data(self):
        merged_primary_df = self.merged_data['region'][(self.get_filter_condition(self.merged_data['region']))]
        merged_secondary_df = self.merged_data['city'][(self.get_filter_condition(self.merged_data['city']))]
        self.data_dict['region']['overall']["top_region"] = merged_primary_df[(merged_primary_df['region'] == self.performer_data['region']['overall']['revenue_growth'])].to_dict(orient="records")
        self.data_dict['region']['overall']["top_region_city"] = merged_secondary_df[(merged_secondary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                                (merged_secondary_df["city"] == self.performer_data['city']['overall']['revenue_growth'])].to_dict(orient="records")
        self.data_dict['region']['overall']['market_leader'] = merged_primary_df[(merged_primary_df['region'] == self.performer_data['region']['overall']['revenue'])].to_dict(orient="records")
        self.data_dict['region']['overall'] = [self.data_dict['region']['overall']]
        self.data_dict['region']['top_performers'] = merged_primary_df[(merged_primary_df['region'].isin(self.performer_data['region']['top_performers']))].sort_values(by='revenue_chg', ascending=False).to_dict(orient="records")
        self.data_dict['region']['bottom_performers'] = merged_primary_df[merged_primary_df['region'].isin(self.performer_data['region']['bottom_performers'])].sort_values(by='revenue_chg', ascending=True).to_dict(orient="records")
        performer_categories = ['top_performers','bottom_performers']
        for performer_category in performer_categories:
            for idx, primary_performer in enumerate(self.data_dict['region'][performer_category]):
                primary_value = primary_performer['region']
                self.data_dict['region'][performer_category][idx]['cities'] = merged_secondary_df[(merged_secondary_df['region'] == primary_value)].sort_values(by='revenue_chg', ascending=False if performer_category=='top_performers' else True).to_dict(orient="records")
        logging.info("Data preparation completed for region, city")

    def prepare_segment_data(self):
        merged_primary_df = self.merged_data['sub_category'][(self.get_filter_condition(self.merged_data['sub_category']))]
        self.data_dict['sub_category']['overall']["top_segment"] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth'])].to_dict(orient="records")
        self.data_dict['sub_category']['overall'] = [self.data_dict['sub_category']['overall']]
        self.data_dict['sub_category']['top_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'].isin(self.performer_data['sub_category']['top_performers']))].sort_values(by='revenue_chg', ascending=False).to_dict(orient="records")
        self.data_dict['sub_category']['bottom_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'].isin(self.performer_data['sub_category']['bottom_performers']))].sort_values(by='revenue_chg', ascending=True).to_dict(orient="records")

        logging.info("Data preparation completed for sub_category")

    def prepare_brand_data(self):
        merged_primary_df = self.merged_data['brand'][(self.get_filter_condition(self.merged_data['brand']))]

        self.data_dict['brand']['overall']["top_brand"] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['brand'] == self.performer_data['brand']['overall']['revenue_growth'])].to_dict(orient="records")
        self.data_dict['brand']['overall'] = [self.data_dict['brand']['overall']]
        self.data_dict['brand']['top_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['brand'].isin(self.performer_data['brand']['top_performers']))].sort_values(by='revenue_chg', ascending=False).to_dict(orient="records")
        self.data_dict['brand']['bottom_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['brand'].isin(self.performer_data['brand']['bottom_performers']))].sort_values(by='revenue_chg', ascending=True).to_dict(orient="records")

        logging.info("Data preparation completed for brand")

    def prepare_sub_brand_data(self):
        merged_primary_df = self.merged_data['sub_brand'][(self.get_filter_condition(self.merged_data['sub_brand']))]

        self.data_dict['sub_brand']['overall']["top_sub_brand"] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['sub_brand'] == self.performer_data['sub_brand']['overall']['revenue_growth'])].to_dict(orient="records")
        self.data_dict['sub_brand']['overall'] = [self.data_dict['sub_brand']['overall']]
        self.data_dict['sub_brand']['top_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_brand'].isin(self.performer_data['sub_brand']['top_performers']))].sort_values(by='revenue_chg', ascending=False).to_dict(orient="records")
        self.data_dict['sub_brand']['bottom_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_brand'].isin(self.performer_data['sub_brand']['bottom_performers']))].sort_values(by='revenue_chg', ascending=True).to_dict(orient="records")

        logging.info("Data preparation completed for sub_brand")

        # with open('data/data_dict.txt', 'w') as json_file:
        #     # 4. Use json.dump() to write the dictionary to the file
        #     # indent=4 makes the file human-readable
        #     json.dump(self.data_dict, json_file, indent=4)
        # with open('data/data_dict.txt', "w") as file:
        #     # Convert the dictionary to a string and write it
        #     file.write(str(self.data_dict))

    def prepare_flavour_data(self):
        merged_primary_df = self.merged_data['flavour'][(self.get_filter_condition(self.merged_data['flavour']))]

        self.data_dict['flavour']['overall']["top_flavour"] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['flavour'] == self.performer_data['flavour']['overall']['revenue_growth'])].to_dict(orient="records")

        self.data_dict['flavour']['overall'] = [self.data_dict['flavour']['overall']]
        self.data_dict['flavour']['top_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['flavour'].isin(self.performer_data['flavour']['top_performers']))].sort_values(by='revenue_chg', ascending=False).to_dict(orient="records")
        self.data_dict['flavour']['bottom_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['flavour'].isin(self.performer_data['flavour']['bottom_performers']))].sort_values(by='revenue_chg', ascending=True).to_dict(orient="records")

        logging.info("Data preparation completed for flavour")

    def prepare_pack_size_data(self):
        merged_primary_df = self.merged_data['pack_size'][(self.get_filter_condition(self.merged_data['pack_size']))]

        self.data_dict['pack_size']['overall']["top_pack_size"] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['pack_size'] == self.performer_data['pack_size']['overall']['revenue_growth'])].to_dict(orient="records")
        self.data_dict['pack_size']['overall'] = [self.data_dict['pack_size']['overall']]

        self.data_dict['pack_size']['top_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['pack_size'].isin(self.performer_data['pack_size']['top_performers']))].sort_values(by='revenue_chg', ascending=False).to_dict(orient="records")
        self.data_dict['pack_size']['bottom_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['pack_size'].isin(self.performer_data['pack_size']['bottom_performers']))].sort_values(by='revenue_chg', ascending=True).to_dict(orient="records")
        logging.info("Data preparation completed for pack size")

    def prepare_pack_group_data(self):
        merged_primary_df = self.merged_data['pack_group'][(self.get_filter_condition(self.merged_data['pack_group']))]

        self.data_dict['pack_group']['overall']["top_pack_group"] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                            (merged_primary_df['pack_group'] == self.performer_data['pack_group']['overall']['revenue_growth'])].to_dict(orient="records")
        self.data_dict['pack_group']['overall'] = [self.data_dict['pack_group']['overall']]

        self.data_dict['pack_group']['top_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['pack_group'].isin(self.performer_data['pack_group']['top_performers']))].sort_values(by='revenue_chg', ascending=False).to_dict(orient="records")
        self.data_dict['pack_group']['bottom_performers'] = merged_primary_df[(merged_primary_df["region"] == self.performer_data['region']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['city'] == self.performer_data['city']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['sub_category'] == self.performer_data['sub_category']['overall']['revenue_growth']) &
                                                                        (merged_primary_df['pack_group'].isin(self.performer_data['pack_group']['bottom_performers']))].sort_values(by='revenue_chg', ascending=True).to_dict(orient="records")
        logging.info("Data preparation completed for pack group")

    def generate_formatted_data(self):
        self.prepare_comparison_time()
        self.merge_data()
        self.performance_ranking()
        self.prepare_country_data()
        self.prepare_region_city_data()
        self.prepare_segment_data()
        self.prepare_brand_data()
        self.prepare_sub_brand_data()
        self.prepare_flavour_data()
        self.prepare_pack_size_data()
        self.prepare_pack_group_data()
        return self.data_dict



# Main entry point for the script
if __name__ == '__main__':
    # pass
    filter_instance = Filter(
        year=2025,
        month=3,
        retailer='LULU',
        comparison='MTD'
    )
    dataframe_creator = DataframeCreator(filter_instance)
    # columns_to_use, group_by_clause = dataframe_creator.get_updated_columns_and_groupby('category')
    # print(columns_to_use)
    with open('Sprint4_MVP1/data/data_dict.txt', "w") as file:
            # Convert the dictionary to a string and write it
            file.write(str(dataframe_creator.generate_formatted_data()))

    # dataframe_creator.prepare_comparison_time()
    # dataframe_creator.merge_data()
    # dataframe_creator.performance_ranking()
    # dataframe_creator.prepare_country_data()
    # dataframe_creator.prepare_region_city_data()
    # dataframe_creator.prepare_segment_data()
    # dataframe_creator.prepare_brand_subbrand_data()
    # dataframe_creator.prepare_flavour_data()
    # dataframe_creator.prepare_pack_group_size_data()
    # dataframe_creator.filter_condition_generator('brand')
