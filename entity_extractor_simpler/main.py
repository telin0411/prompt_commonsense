import pandas as pd

if __name__ == '__main__':
    df = pd.read_json('./datasets/com2sense/com2sense.json')
    cnt = 0
    for index, row in df.iterrows():
        cnt += 1
    print(cnt)
