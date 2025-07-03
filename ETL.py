import pandas as pd 
from dataclasses import dataclass 
import boto3, sqlite3, os, yaml
from botocore.exceptions import ClientError


@dataclass 
class ETL:
    database_name:str 
    data:pd.DataFrame 
    table_name: str 
    cols:dict[str, str | list[str] | tuple[str]]
    file_path:str 
    bucket:str 
    object_name:str 
    
    def create_database(self) -> None:
        con = sqlite3.connect(self.database_name)
        con.close()
        print(f"created database \'{self.database_name}\'")

    def create_table(self, table_name:str, cols:dict[str, str | list[str] | tuple[str]]):
        """creates table inside database

        Args:
            table_name (str): name of the table
            cols (dict[str, str | list[str] | tuple[str]]): 
                - key --> column name, 
                - value --> list of constrains, data type, default value etc.
        """
        connection = sqlite3.connect(self.database_name)
        cursor = connection.cursor()
        string = ""
        lenght = 0
        for key, value in cols.items():
            lenght += 1
            base_val = ""
            count = 0
            if isinstance(value, list) or isinstance(value, tuple):
                for val in value:
                    if count == 0:
                        base_val = val
                        count+=1
                    else:
                        base_val = base_val + " " + val 
            else:
                base_val = value 
            if lenght != len(cols.keys()):
                string = string+key+" "+base_val+", \n"
            else:
                string = string+key+" "+base_val
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} ({string})''')
        connection.commit()
        connection.close()
        print(f"Table \'{table_name}\' created successfully.")

    def insert_data(self, table_name:str, records:pd.DataFrame):
        connection = sqlite3.connect(self.database_name)
        cursor = connection.cursor()
        cursor.executemany(f"""
            INSERT INTO {table_name} {tuple(records.columns)} VALUES 
                    ({",".join(["? " for _ in range(len(records.columns))])})
        """, list(zip(*[records[col] for col in records.columns]))
        )
        connection.commit()
        connection.close()
        print(f"{records.shape[0]} records inserted.")
    
    def push_to_cloud(self):
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(self.file_path, self.bucket, self.object_name)
        except ClientError as e:
            raise e
    
    def save_schema(self)->None:
        """saves the schema data
        """
        path = "schema"
        os.makedirs(path, exist_ok=True)

        schema = dict()
        columns_with_dtype = dict()
        numerical_columns = list()

        for col in self.data.columns:
            columns_with_dtype[col] = str(self.data[col].dtype)
            if self.data[col].dtype!="O":
                numerical_columns.append(col)

        schema["columns"] = columns_with_dtype
        schema["numerical_columns"] = numerical_columns

        with open(os.path.join(path, "schema.yaml"), "w") as file:
            yaml.safe_dump(schema, file)

    def main(self):
        self.create_database()
        self.create_table(self.table_name, self.cols)
        self.insert_data(self.table_name, self.data)
        self.push_to_cloud()
        self.save_schema()


if __name__ == "__main__":
    os.makedirs("database", exist_ok=True)
    file_path = "database/Bank.db"
    data_path = "D:/MyDatasets/ChurnModelling/data.csv"
    obj = ETL(
        file_path,
        pd.read_csv(data_path),
        "ChurnModelling",
        {
            'RowNumber': 'INTEGER',
            'CustomerId': 'INTEGER',
            'Surname': 'TEXT',
            'CreditScore': 'INTEGER',
            'Geography': 'TEXT',
            'Gender': 'TEXT',
            'Age': 'INTEGER',
            'Tenure': 'INTEGER',
            'Balance': 'REAL',
            'NumOfProducts': 'INTEGER',
            'HasCrCard': 'INTEGER',
            'IsActiveMember': 'INTEGER',
            'EstimatedSalary': 'REAL',
            'Exited': 'INTEGER'
        },
        file_path,
        'projectsbucket01', 
        'ChurnModelling-db'
    )
    obj.main()

