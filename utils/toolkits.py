#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import pprint
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yaml


def load_config_yml(config_file):
    with open(config_file, 'r') as yml_file:
        configs = yaml.load(yml_file, Loader=yaml.FullLoader)
    return configs


class gsio(object):
    def __init__(self, credential, sheet_name):
        # use creds to create a client to interact with the Google Drive API
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            credential, scope)
        self.client = gspread.authorize(creds)
        self.sheet_name = sheet_name

    def read_sheet(self, sheet_title):
        # Find a workbook by name and open a sheet
        # Make sure you use the right name here.
        sheet = self.client.open(self.sheet_name).worksheet(sheet_title)

        # Extract and print all of the values
        list_of_hashes = sheet.get_all_records()
        #pprint.pprint(list_of_hashes)
        return list_of_hashes

    def insert_row(self, sheet_title, row_data):
        """
        insert a row into a sheet
        :param sheet_title: the title of sheet
        :param row_data: a list of row data
        :return:
        """
        sheet_obj = self.client.open(self.sheet_name).worksheet(sheet_title)
        sheet_obj.append_row(values=row_data)


if __name__ == "__main__":
    credential = 'credentials.json'
    sheet_name = "name"
    gs = gsio(credential, sheet_name)
    gs.read_sheet('sheet_1')
    data = {'date': '2018-10-11', 'model': 'cnn',
            'acc_train': 0.5, 'acc_val': 0.5, 'acc_test': 0.5}
    data_list = [data['date'], data['model'],
                 data['acc_train'], data['acc_val'], data['acc_test']]
    gs.insert_row(sheet_title='sheet_1', row_data=data_list)
