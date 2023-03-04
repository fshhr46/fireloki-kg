import numpy as np
import pandas as pd


def process_sex(df):
    def helper(sex: str):
        if sex == 'male':
            return 1
        elif sex == 'female':
            return 0
        else:
            return -1
    sex_feat = df['Sex'].map(helper)
    df['SexDigit'] = sex_feat


def process_age(df):
    def helper(age: float):
        if pd.isna(age):
            return -1
        return age
    age_feat = df['Age'].map(helper)
    df['Age'] = age_feat



def process_fare(df):
    def helper(fare: float):
        if pd.isna(fare):
            return 0
        return fare
    fare_feat = df['Fare'].map(helper)
    df['Fare'] = fare_feat

def process_embarked(df):
    def helper(embarked: str):
        valid_codes = 'CQS'
        if pd.isna(embarked):
            return -1
        else:
            return valid_codes.index(embarked)
    embarked_feat = df['Embarked'].map(helper)
    df['EmbarkedDigit'] = embarked_feat

def process_cabin(df):

    unique_cabins = df.loc[:, "Cabin"].unique()

    # remove nan if exists
    if pd.isna(unique_cabins[0]):
        unique_cabins = unique_cabins[1:]

    cbin_to_idx = {}
    for idx, cbin_str in enumerate(unique_cabins):
        cbin_to_idx[cbin_str] = idx

    def encode_cbin(cbin_str: str):
        if pd.isna(cbin_str):
            return -1, -1, -1

        cbins = cbin_str.split(" ")
        if len(cbins) > 1:
            cbin_code, cbin_num = -1, -1
        else:
            cbin_str = cbins[0]
            cbin_code, cbin_num = ord(cbin_str[0]) - ord("A"), cbin_str[1:]

            # Validate cbin_num
            if not (0 <= cbin_code < 26):
                cbin_code = -1

            # Validate cbin_num
            if cbin_num == '':
                cbin_num = -1
            else:
                cbin_num = int(cbin_num)
        return cbin_code, cbin_num, cbin_to_idx[cbin_str]

    feats = df['Cabin'].map(encode_cbin)
    df['CabinCode'] = feats.map(lambda x:x[0])
    df['CabinNum'] = feats.map(lambda x:x[1])
    df['CabinIndex'] = feats.map(lambda x:x[2])

def process_ticket(df):

    def tokenize_tt_str(tt_str: str):

        if pd.isna(tt_str):
            return tt_str

        tt_str = tt_str.strip()
        tokens = tt_str.split(" ")
        prefix, number = tokens[:-1], tokens[-1]

        # number only
        if not prefix:
            return "", number
        else:
            prefix = ''.join(prefix)
            prefix = prefix.replace(".", "")
            return prefix, number


    unique_tt = df.loc[:, "Ticket"].unique()
    # remove nan if exists
    if pd.isna(unique_tt[0]):
        unique_tt = unique_tt[1:]

    unique_tt_normalized = set()
    unique_prefix = set()
    unique_nums = set()
    for tt_str in unique_tt:
        prefix, number = tokenize_tt_str(tt_str)
        unique_tt_normalized.add(f"{prefix} {number}")
        unique_prefix.add(prefix)

    prefix_to_idx = {}
    for idx, prefix in enumerate(unique_prefix):
        prefix_to_idx[prefix] = idx

    def encode_ticket(tt_str: str):
        if pd.isna(tt_str):
            tt_type = -1

        prefix, number = tokenize_tt_str(tt_str)

        # number only
        # if not prefix:
        #     if number.isnumeric():
        #         tt_type = 0
        #     else:
        #         tt_type = 1
        # else:
        #     prefix_tokens = prefix[0].split('/')
        #     if len(prefix_tokens) == 1:
        #         tt_type = 2
        #     tt_type = 3
        tt_type = prefix_to_idx[prefix]
        return tt_type, f"{prefix} {number}"

    feats = df['Ticket'].map(encode_ticket)
    df['TicketDigit'] = feats.map(lambda x:x[0])
    df['TicketNormalized'] = feats.map(lambda x:x[1])

def extract_feature(df):
    process_age(df)

    process_fare(df)

    process_sex(df)

    process_embarked(df)

    process_cabin(df)

    process_ticket(df)

    return df
