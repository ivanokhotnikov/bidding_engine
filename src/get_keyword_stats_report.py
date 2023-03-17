#!/usr/bin/env python
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example illustrates how to get campaign criteria.
Retrieves negative keywords in a campaign.
see https://github.com/googleads/google-ads-python/blob/b6a5dffccc25872fd38925926e59988186e3cc15/examples/reporting/get_keyword_stats.py
"""

import argparse
import csv
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException


# [START get_keyword_stats]
def main(client, customer_id):
    ga_service = client.get_service('GoogleAdsService')

    query = """
        SELECT
          campaign.id,
          campaign.name,
          ad_group.id,
          ad_group.name,
          ad_group_criterion.criterion_id,
          ad_group_criterion.keyword.text,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros,
          metrics.absolute_top_impression_percentage,
          metrics.top_impression_percentage,
          metrics.search_rank_lost_top_impression_share,
          metrics.search_top_impression_share,
          metrics.search_impression_share
        FROM keyword_view WHERE segments.date DURING LAST_7_DAYS
        AND campaign.advertising_channel_type = 'SEARCH'
        AND ad_group.status = 'ENABLED'
        AND ad_group_criterion.status IN ('ENABLED', 'PAUSED')
        ORDER BY metrics.impressions DESC
        LIMIT 50"""

    # Issues a search request using streaming.
    search_request = client.get_type('SearchGoogleAdsStreamRequest')
    search_request.customer_id = customer_id
    search_request.query = query
    stream = ga_service.search_stream(search_request)
    with open(
            os.path.join(os.environ['RAW_DATA_PATH'],
                         datetime.now().strftime('%Y%m%d%H%M%S')), 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow([
            'CampaignId', 'CampaignName', 'AdGroupId', 'AdGroupName',
            'CriterionId', 'Criteria', 'Impressions', 'Clicks', 'Cost',
            'AbsoluteTopImpressionPercentage', 'TopImpressionPercentage',
            'SearchRankLostTopImpressionShare', 'SearchTopImpressionShare',
            'SearchImpressionShare'
        ])
        for batch in stream:
            for row in batch.results:
                campaign = row.campaign
                ad_group = row.ad_group
                criterion = row.ad_group_criterion
                metrics = row.metrics
                writer.writerow([
                    campaign.id, campaign.name, ad_group.id, ad_group.name,
                    criterion.criterion_id, criterion.keyword.text,
                    metrics.impressions, metrics.clicks, metrics.cost_micros,
                    metrics.absolute_top_impression_percentage,
                    metrics.top_impression_percentage,
                    metrics.search_rank_lost_top_impression_share,
                    metrics.search_top_impression_share,
                    metrics.search_impression_share
                ])
    # [END get_keyword_stats]


if __name__ == '__main__':
    # GoogleAdsClient will read the google-ads.yaml configuration file in the
    # home directory if none is specified.
    load_dotenv()
    googleads_client = GoogleAdsClient.load_from_storage(
        os.path.join(os.environ['CONF_PATH'], 'google-ads.yaml'))
    parser = argparse.ArgumentParser(
        description=("Retrieves a campaign's negative keywords."))
    # The following argument(s) should be provided to run the example.
    parser.add_argument('-c',
                        '--customer_id',
                        type=str,
                        required=True,
                        help='The Google Ads customer ID.')
    args = parser.parse_args()

    try:
        main(googleads_client, args.customer_id)
    except GoogleAdsException as ex:
        print(f'Request with ID "{ex.request_id}" failed with status '
              f'"{ex.error.code().name}" and includes the following errors:')
        for error in ex.failure.errors:
            print(f'\tError with message "{error.message}".')
            if error.location:
                for field_path_element in error.location.field_path_elements:
                    print(f'\t\tOn field: {field_path_element.field_name}')
        sys.exit(1)
