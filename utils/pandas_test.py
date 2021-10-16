import pandas as pd


class PandasTest:
    def test_pandas_csv(self, path: str):
        # df = pd.read_csv('test.csv')
        df = pd.read_csv(path)
        print('head', df.head())


p = PandasTest()
p.test_pandas_csv('/Users/zhangshipeng/PycharmProjects/ephemeralP/utils/test.csv')
