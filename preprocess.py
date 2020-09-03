import pandas as pd
import numpy as np
import pickle
import sys

def negative_sampling(triplet_df, user_list, item_list, 
                      brand_list, entity_list, entity_type, relation_type):
    # 比率を考える
    pos_triplet = [list(row) for row in triplet_df.values]
    nega_triplet = []

    relation_list = triplet_df['relation']
    dice = [len(relation_list[relation_list == i]) / len(relation_list) for i in range(len(relation_type))]
    sampling_relation_num = np.random.multinomial(len(triplet_df), dice) 
    for i in range(len(sampling_relation_num)):
        relation = i
        count = 0
        while count < sampling_relation_num[i]:
            h_entity = np.random.randint(len(entity_list))
            t_entity = np.random.randint(len(entity_list))
            if [h_entity, t_entity, relation] in pos_triplet:
                continue
            if [h_entity, t_entity, relation] in nega_triplet:
                continue

            nega_triplet.append([h_entity, t_entity, relation])
            count += 1

    nega_triplet_df = pd.DataFrame(nega_triplet, columns = ['h_entity', 't_entity', 'relation'])

    #保存
    #nega_triplet_df.to_csv('./data/nega_triplet.csv', index=False)

    return nega_triplet_df
    

def negative_sampling_bpr(train_df, user_list, item_list):
    # implicit_feed = [list(r) for r in user_item_df.values]
    implicit_feed = [list(r) for r in train_df.values]

    user_item_train_nega = []

    count = 0
    #while count < 1000:
    while count < len(train_df):
        #user = user_list[np.random.randint(user_num)]
        #item = item_list[np.random.randint(item_num)]
        user = np.random.randint(len(user_list))
        item = np.random.randint(len(item_list))
        if [user, item] in implicit_feed:
            continue
        if [user, item] in user_item_train_nega:
            continue

        user_item_train_nega.append([user, item])
        count += 1

    user_item_train_nega_df = pd.DataFrame(user_item_train_nega, columns=['reviewerID', 'asin'])
    #保存
    #user_item_train_nega_df.to_csv('./data/bpr/user_item_train_nega.csv', index=False)

    return user_item_train_nega_df
    
    

def mk_triplet(train_df):
    # 一つのtriplet dataframeを作る
    # これが訓練データになる
    # e_1, e_2, relation　が行
    triplet_df = []
    for row in train_df.values:
        user = entity_list.index(row[0])
        item = entity_list.index(row[1])
        triplet_df.append([user, item, relation_type.index('u_buy_i')])

    for row in item_brand_df.values:
        if row[0] not in entity_list:
            continue
        if row[1] not in entity_list:
            continue
        item = entity_list.index(row[0])
        brand = entity_list.index(row[1])
        triplet_df.append([item, brand, relation_type.index('i_belong_b')])

    for row in item_buy_item_df.values:
        if row[0] not in entity_list:
            continue
        item_id = entity_list.index(row[0])
        if type(row[1]) != str:
            continue
        also_i = row[1][1:-1].split(',')
        if len(also_i) == 0:
            continue

        for a_i in also_i:
            #print(a_i)
            if a_i[1:-1] not in entity_list: continue
            also_item_id = entity_list.index(a_i[1:-1])
            triplet_df.append([item_id, also_item_id, relation_type.index('i_also_buy_i')])

    for row in item_view_item_df.values:
        if row[0] not in entity_list:
            continue
        item_id = entity_list.index(row[0])
        if type(row[1]) != str:
            continue
        also_i = row[1][1:-1].split(',')
        if len(also_i) == 0:
            continue

        for a_i in also_i:
            #print(a_i)
            if a_i[1:-1] not in entity_list: continue
            also_item_id = entity_list.index(a_i[1:-1])
            triplet_df.append([item_id, also_item_id, relation_type.index('i_also_view_i')])

    triplet_df = pd.DataFrame(triplet_df, columns=['h_entity', 't_entity', 'relation'])

    return triplet_df


# データに含まれるuser-item1, item2, item3, ...を返す
# 辞書
def user_aggregate_item(df):
    user_items_dict = {}
    #for user in user_list:
    for i in range(len(item_list), len(item_list) + len(user_list)):
        items_df = df[df['reviewerID'] == i]
        user_items_dict[i] = list(items_df['asin'])
    return user_items_dict

def user_aggregate_item_bpr(df):
    user_items_dict = {}
    #for user in user_list:
    for i in range(len(user_list)):
        items_df = df[df['reviewerID'] == i]
        user_items_dict[i] = list(items_df['asin'])
    return user_items_dict

def id_test_df(test_df):
    # user_item_test_dfをID化する
    user_item_test = []
    user_item_test_bpr = []
    for row in test_df.values:
        user = entity_list.index(row[0])
        item = entity_list.index(row[1])
        # BPR用
        user_bpr = user_list.index(row[0])
        item_bpr = item_list.index(row[1])
        user_item_test.append([user, item, relation_type.index('u_buy_i')])
        user_item_test_bpr.append([user_bpr, item_bpr])

    user_item_test_df = pd.DataFrame(user_item_test, columns = ['reviewerID', 'asin', 'relation'])
    user_item_test_bpr_df = pd.DataFrame(user_item_test_bpr, columns = ['reviewerID', 'asin'])

    return user_item_test_df, user_item_test_bpr_df

def id_train_bpr_df(train_df):
    # user_item_train_dfをBPR用にID化する
    user_item_train = []
    for row in train_df.values:
        user = user_list.index(row[0])
        item = item_list.index(row[1])
        user_item_train.append([user, item])
    user_item_train_bpr_df = pd.DataFrame(user_item_train, columns = ['reviewerID', 'asin'])
    return user_item_train_bpr_df

def mk_valid(dir, train_df, test_df):

    # test_dfをid化
    _test_df, _test_bpr_df = id_test_df(test_df)
    #保存
    _test_df.to_csv(dir + 'user_item_test.csv', index=False)
    _test_bpr_df.to_csv(dir + 'bpr/user_item_test.csv', index=False)

    # BPR用のtrain_dfのID化
    _train_bpr_df = id_train_bpr_df(train_df)
    #保存
    _train_bpr_df.to_csv(dir + 'bpr/user_item_train.csv', index=False)


    # tripletを作る
    _triplet_df = mk_triplet(train_df)
    _triplet_df.to_csv(dir + 'triplet.csv', index=False)


    # negative sampling
    _nega_triplet_df = negative_sampling(_triplet_df, user_list, item_list, 
                      brand_list, entity_list, entity_type, relation_type)

    _user_item_train_nega_bpr_df = negative_sampling_bpr(_train_bpr_df, user_list,
                                       item_list)
    _nega_triplet_df.to_csv(dir + 'nega_triplet.csv', index=False)
    _user_item_train_nega_bpr_df.to_csv(dir + 'bpr/user_item_train_nega.csv', index=False)


    # trainデータに対するtargetを作る
    y_train = np.array([1 for i in range(len(_triplet_df))] + [0 for i in range(len(_nega_triplet_df))])
    #保存
    np.savetxt(dir + '/y_train.txt', y_train)


    # user aggregate item
    _test_dict = user_aggregate_item(_test_df)
    _nega_bpr_dict = user_aggregate_item_bpr(_user_item_train_nega_bpr_df)
    _test_bpr_dict = user_aggregate_item_bpr(_test_bpr_df)

    with open(dir + 'user_items_test_dict.pickle', 'wb') as f:
        pickle.dump(_test_dict, f)

    with open(dir + 'bpr/user_items_nega_dict.pickle', 'wb') as f:
        pickle.dump(_nega_bpr_dict, f)
    
    with open(dir + 'bpr/user_items_test_dict.pickle', 'wb') as f:
        pickle.dump(_test_bpr_dict, f)



def mk_es_valid(dir, train_df):


    # BPR用のtrain_dfのID化
    _train_bpr_df = id_train_bpr_df(train_df)
    _train_bpr_df.to_csv(dir + 'bpr/user_item_train.csv', index=False)

    # tripletを作る
    _triplet_df = mk_triplet(train_df)
    _triplet_df.to_csv(dir + 'triplet.csv', index=False)


    # negative sampling
    _nega_triplet_df = negative_sampling(_triplet_df, user_list, item_list, 
                      brand_list, entity_list, entity_type, relation_type)

    _user_item_train_nega_bpr_df = negative_sampling_bpr(_train_bpr_df, user_list,
                                       item_list)
    _nega_triplet_df.to_csv(dir + 'nega_triplet.csv', index=False)
    _user_item_train_nega_bpr_df.to_csv(dir + 'bpr/user_item_train_nega.csv', index=False)


    # trainデータに対するtargetを作る
    y_train = np.array([1 for i in range(len(_triplet_df))] + [0 for i in range(len(_nega_triplet_df))])
    #保存
    np.savetxt(dir + '/y_train.txt', y_train)


    # user aggregate item
    _nega_bpr_dict = user_aggregate_item_bpr(_user_item_train_nega_bpr_df)
    
    with open(dir + 'bpr/user_items_nega_dict.pickle', 'wb') as f:
        pickle.dump(_nega_bpr_dict, f)




if __name__ == '__main__':
    args = sys.argv

    # 保存ディレクトリ
    #dir_test = './data_luxury_5core/test/'
    #dir_valid1 = './data_luxury_5core/valid1/'
    #dir_valid2 = './data_luxury_5core/valid2/'
    dir_test = args[1] + '/test/'
    dir_valid1 = args[1] + '/valid1/'
    dir_valid2 = args[1] + '/valid2/'

    # データ読み込み
    #user_item_df = pd.read_csv('./Luxury_5core/user_item.csv')
    #item_brand_df = pd.read_csv('./Luxury_5core/item_brand.csv')
    #item_buy_item_df = pd.read_csv('./Luxury_5core/item_buy_item.csv')
    #item_view_item_df = pd.read_csv('./Luxury_5core/item_view_item.csv')
    user_item_df = pd.read_csv(args[2] + '/user_item.csv')
    item_brand_df = pd.read_csv(args[2] + '/item_brand.csv')
    item_buy_item_df = pd.read_csv(args[2] + '/item_buy_item.csv')
    item_view_item_df = pd.read_csv(args[2] + '/item_view_item.csv')

    entity_type = ['user', 'item', 'brand']
    relation_type = ['u_buy_i', 'i_belong_b', 'i_also_buy_i', 'i_also_view_i']

    # 各entity_typeのリストを作る
    item_list = list(set(list(user_item_df['asin'])))
    user_list = list(set(list(user_item_df['reviewerID'])))
    brand_list = list(set(list(item_brand_df['brand'])))
    # nanを除く
    brand_list.pop(0)

    print('item {}'.format(len(item_list)))
    print('user {}'.format(len(user_list)))
    print('brand {}'.format(len(brand_list)))

    # 保存
    with open(dir_test + 'user_list.txt', 'w') as f:
        for user in user_list:
            f.write(user + '\n')
    with open(dir_test + 'item_list.txt', 'w') as f:
        for item in item_list:
            f.write(item + '\n')
    with open(dir_test + 'brand_list.txt', 'w') as f:
        for brand in brand_list:
            f.write(brand + '\n')

    # entityのリストを一つに連結する
    # このリストを使ってentityのidxを管理
    entity_list = item_list + user_list + brand_list
    print('entity size: {}'.format(len(entity_list)))

    # 保存
    with open(dir_test + 'entity_list.txt', 'w') as f:
        for entity in entity_list:
            f.write(entity + '\n')


    # user-itemインタラクションをスプリットする
    # Early Stoppingようのvalidデータをつくる
    user_item_df = user_item_df.take(np.random.permutation(len(user_item_df)))
    train_num = int(3/4 * len(user_item_df))
    user_item_train_df = user_item_df[0:train_num]
    valid_num = int(1/3 * len(user_item_train_df))
    user_item_valid_df = user_item_train_df[0:valid_num]
    user_item_train_df = user_item_train_df[valid_num:]
    user_item_test_df = user_item_df[train_num:]

    print('train num {}'.format(len(user_item_train_df)))
    print('vaid num {}'.format(len(user_item_valid_df)))
    print('test num {}'.format(len(user_item_test_df)))

    #print('train {}'.format(train_num))
    #print('test {}'.format(len(user_item_test_df)))

    # early stopping用にデータを作る
    mk_es_valid(args[1] + '/early_stopping/', user_item_valid_df)

    # 2cross validデータを作る
    valid_num = int(0.5 * len(user_item_train_df))
    mk_valid(dir_valid1, user_item_train_df[:valid_num], user_item_train_df[valid_num:])
    mk_valid(dir_valid2, user_item_train_df[valid_num:], user_item_train_df[:valid_num])

    # test_dfをid化
    user_item_test_df, user_item_test_bpr_df = id_test_df(user_item_test_df)
    #保存
    user_item_test_df.to_csv(dir_test + 'user_item_test.csv', index=False)
    user_item_test_bpr_df.to_csv(dir_test + 'bpr/user_item_test.csv', index=False)

    # BPR用のtrain_dfのID化
    user_item_train_bpr_df = id_train_bpr_df(user_item_train_df)
    #保存
    user_item_train_bpr_df.to_csv(dir_test + 'bpr/user_item_train.csv', index=False)


    # tripletを作る
    _triplet_df = mk_triplet(user_item_train_df)
    _triplet_df.to_csv(dir_test + 'triplet.csv', index=False)


    # negative sampling
    nega_triplet_df = negative_sampling(_triplet_df, user_list, item_list, 
                      brand_list, entity_list, entity_type, relation_type)

    user_item_train_nega_bpr_df = negative_sampling_bpr(user_item_train_bpr_df, user_list,
                                       item_list)
    nega_triplet_df.to_csv(dir_test + 'nega_triplet.csv', index=False)
    user_item_train_nega_bpr_df.to_csv(dir_test + 'bpr/user_item_train_nega.csv', index=False)


    # trainデータに対するtargetを作る
    y_train = np.array([1 for i in range(len(_triplet_df))] + [0 for i in range(len(nega_triplet_df))])
    #保存
    np.savetxt(dir_test + 'y_train.txt', y_train)


    # user aggregate item
    user_items_test_dict = user_aggregate_item(user_item_test_df)
    user_items_nega_bpr_dict = user_aggregate_item_bpr(user_item_train_nega_bpr_df)
    user_items_test_bpr_dict = user_aggregate_item_bpr(user_item_test_bpr_df)

    with open(dir_test + 'user_items_test_dict.pickle', 'wb') as f:
        pickle.dump(user_items_test_dict, f)

    with open(dir_test + 'bpr/user_items_nega_dict.pickle', 'wb') as f:
        pickle.dump(user_items_nega_bpr_dict, f)
    
    with open(dir_test + 'bpr/user_items_test_dict.pickle', 'wb') as f:
        pickle.dump(user_items_test_bpr_dict, f)
