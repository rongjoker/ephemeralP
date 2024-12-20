import os
import xlrd
import pandas as pd


def read_by_slrd():
  wb = xlrd.open_workbook('线边仓仓库库位明细.xlsx')
  #按工作簿定位工作表
  sh = wb.sheet_by_name('Sheet3')
  print(sh.nrows)#有效数据行数
  print(sh.ncols)#有效数据列数
  print(sh.cell(0,0).value)#输出第一行第一列的值
  print(sh.row_values(0))#输出第一行的所有值
  #将数据和标题组合成字典
  print(dict(zip(sh.row_values(0),sh.row_values(1))))
  #遍历excel，打印所有数据
  for i in range(sh.nrows):
      print(sh.row_values(i))


def read_by_pd(table_name,file_name):
  df1 = pd.read_excel('线边仓仓库库位明细.xlsx',
                      sheet_name='Sheet3',
                      header=0,
                      # engine='openpyxl',
                      )
  print(df1.head())
  print('-----------------')
  print(df1['AREANO'])
  print('-----------------')
  print(df1['WAREHOUSENO'])

  # 生成SQL语句
  update_statements = []
  insert_statements = []
  insert_statements_duplicate = []

  for _, row in df1.iterrows():
    areano = row['AREANO']
    warehouseno = row['WAREHOUSENO']

    # 更新语句
    update_sql = f"""
      UPDATE {table_name}
      SET cd_warehouse_code = '{warehouseno}'
      WHERE location_no = '{areano}';
      """
    update_statements.append(update_sql.strip())

    # 插入语句
    insert_sql = f"""
      INSERT INTO {table_name} (location_no, cd_warehouse_code, enabled, is_deleted, create_at)
      SELECT '{areano}', '{warehouseno}', 1, 0, NOW()
      WHERE NOT EXISTS (
          SELECT 1 FROM {table_name} WHERE location_no = '{areano}'
      );
      """
    insert_statements.append(insert_sql.strip())
    # 插入重复
    insert_sql_d = f"""
      INSERT INTO {table_name} (location_no, cd_warehouse_code, enabled, is_deleted, create_at)
      SELECT '{areano}', '{warehouseno}', 1, 0, NOW()
      WHERE NOT EXISTS (
          SELECT 1 FROM {table_name} WHERE location_no = '{areano}' and cd_warehouse_code = '{warehouseno}'
      );
      """
    insert_statements_duplicate.append(insert_sql_d.strip())

  # 合并更新和插入语句
  output_sql = "\n".join(update_statements + insert_statements + insert_statements_duplicate)
  # print(output_sql)
  # 保存到文件
  # file_name = 'insert_update_location.sql'
  with open(file_name + '.sql', 'w') as file:
    file.write(output_sql + '')

  with open(file_name + '_update.sql', 'w') as file:
    file.write("\n".join(update_statements))

  with open(file_name + '_insert.sql', 'w') as file:
    file.write("\n".join(insert_statements))

  with open(file_name + '_insert_d.sql', 'w') as file:
    file.write("\n".join(insert_statements_duplicate))

  print(f"SQL statements have been saved to {file_name}")

read_by_pd('bas_location_bk','insert_update_location_bk')