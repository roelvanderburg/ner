"""This module collects features from other data sources"""
import requests
import urllib3
from typing import Dict
# hide InsecureRequestWarning todo: fix this warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_name_address(bsn : int) -> Dict:
    """
    Retrieves the name and address information given a bsn
    :param bsn:  bsn number
    :return: response request dictionary from database
    """

    # set host/service/wsa_from
    wsa_from = 'http://servicespecifications.belastingdienst.nl/iva/wca'
    
    # url = 'https://{host}/{service}/{finr}'.format(host=config.get('mdm.host'), service=config.get('mdm.service_address'), finr=finr)
    url = 'https://iva-mihproxyservice.belastingdienst.nl/iva-mihproxyservice/rest/mihproxy/v1/person/{}'.format(bsn)
    headers = {
        'accept': 'application/json',
        'X_WSA_ADDRESS_X': wsa_from
    }

    # make the call to mdm rest api
    response = requests.get(url, headers=headers, verify=False)

    # if bad request, return None
    # if response.status_code != 200:
    #     print( "bad request ")
    #     return None

    response_data = response.json()
    return {'personNames': response_data['personNames']}

