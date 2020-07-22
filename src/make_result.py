import pandas as pd
from collections import Counter

# result_1 = pd.read_table(
#     '/home/ps/Projects/sunspot/src/result_0722_c0.1.txt',
#     sep=' ',
#     header=0)

# result_3 = pd.read_table(
#     '/home/ps/Projects/sunspot/src/result_0722_c0.3.txt',
#     sep=' ',
#     header=0)

# result_5 = pd.read_table(
#     '/home/ps/Projects/sunspot/src/result_0722_c0.5.txt',
#     sep=' ',
#     header=0)

# result_7 = pd.read_table(
#     '/home/ps/Projects/sunspot/src/result_0722_c0.7.txt',
#     sep=' ',
#     header=0)

# result_9 = pd.read_table(
#     '/home/ps/Projects/sunspot/src/result_0722_c0.9.txt',
#     sep=' ',
#     header=0)

# df_r = pd.merge(result_1, result_3, on='id')
# df_r = pd.merge(df_r, result_5, on='id')
# df_r = pd.merge(df_r, result_7, on='id')
# df_r = pd.merge(df_r, result_9, on='id')
# print(df_r)
# df_r.to_csv(
#     "result_0722_all_model.csv",
#     header=False
# )
a = 0
b = 0
with open("/home/ps/Projects/sunspot/src/result_0722_all_model.csv", 'r') as a_file:
    with open("push_up.txt", "a") as push_up:
        lines = a_file.readlines()
        for line in lines:
            line = line.strip('\n')
            item_list = line.split(',')[1:]
            print(item_list)
            item_id = item_list[0]
            result_list = item_list[1:]
            top_1 = Counter(result_list).most_common(1)
            item_id = item_id.zfill(6)
            if top_1[0][1] < 3:
                print(item_id)
                print(top_1)
                a += 1
                # push_up.write("{} {}\n".format(item_id, result_list[2]))
            else:
                # push_up.write("{} {}\n".format(item_id, top_1[0][0]))
                pass
print(a)
