import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./com2sense.json')
    print(len(df[0]))
