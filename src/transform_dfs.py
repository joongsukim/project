import pandas as pd
import os
from typing import List

CO2_raw_data_root = "..\\data\\CO2\\raw\\test"

#"data/CO2/raw/train"
#"data/CO2/raw/val"
#"data/CO2/raw/test"


class CO2DataFrameLoader:
    def __init__(
            self,
            dataset_dir: str
    ):
        super().__init__()
        self.dataset_dir = dataset_dir

    def enumerate_xls_files(self):
        """

        Returns: [Excel Files Path String]
        """
        xls_files = []

        # self.dataset_dir 안의 모든 파일 중에서
        for file_name in os.listdir(self.dataset_dir):
            file_path = os.path.join(self.dataset_dir, file_name)

            # 확장자가 csv인 파일을 csv_files에 모음
            if file_path.endswith('.xlsx'):
                xls_files.append(file_path)

        # csv_files를 알파벳 순서대로 정렬
        xls_files.sort()

        # csv 파일 경로 리스트 반환
        return xls_files

    # TODO: implement later
    def transform(self) -> None:
        xls_files = self.enumerate_xls_files()

        for i, xls_path in enumerate(xls_files):
            print(f"[{i+1}/{len(xls_files)}")
            df1 = pd.read_excel(xls_path, sheet_name=0, header=[0, 1], index_col=0)
            df2 = pd.read_excel(xls_path, sheet_name=1, header=[0, 1], index_col=0)
            df3 = pd.read_excel(xls_path, sheet_name=2, header=[0, 1], index_col=0)

            df = df1.join(df2, how='inner')
            df = df.join(df3, how='inner')

            df.drop(['AVG', 'MAX', 'MIN', 'STD'], inplace=True)
            df.sort_index(inplace=True)

            target_path = xls_path + '.pkl'
            df.to_pickle(target_path)


if __name__ == "__main__":
    loader = CO2DataFrameLoader(CO2_raw_data_root)
    loader.transform()

