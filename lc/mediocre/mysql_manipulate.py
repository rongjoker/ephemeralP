import pymysql


def __get_db__():
    db = pymysql.connect(
        host="rds40-mysql-shinho-datalake-dataplatform-dev.c0dvigf5kjq0.rds.cn-north-1.amazonaws.com.cn",
        user="dl_dev",
        password="Dldev#$%20190912",
        port=3306,
        database="aio",
        charset='utf8')

    return db


def __get_cursor__():
    db = __get_db__()
    return db.cursor()


def query():
    print(' start query ...')
    # SQL 查询语句
    sql = "SELECT uid,task_type,module,create_date FROM fdt_task_flow WHERE uid > {} limit 10".format(1000)
    try:
        cursor = __get_cursor__()
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        for row in results:
            uid = row[0]
            task = row[1]
            module = row[2]
            create_date = row[3]
            # 打印结果
            print("uid={},task={},module={},create_date={}".format(uid, task, module, create_date))
    except Exception as x:
        print("Error: unable to fecth data")
        print(x)

        print(' start end ...')


def query_one(num):
    print(f'num is {num}')
    # SQL 查询语句
    sql = "SELECT uid,task_type,module,create_date,message FROM fdt_task_flow WHERE uid = {} ".format(num)
    try:
        cursor = __get_cursor__()
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        for row in results:
            uid = row[0]
            task = row[1]
            module = row[2]
            create_date = row[3]
            message = row[4]
            # 打印结果 print("uid={},task={},module={},message={},create_date={}".format(uid, task, module, message,
            # create_date))
            return Customer(uid, task, module, create_date, message)

    except Exception as x:
        print("Error: unable to fecth data")
        print(x)

        print(' start end ...')


def add(task_type: 'test from python'):
    sql = "insert into fdt_task_flow(task_type) values({})".format(task_type)
    try:
        cursor = __get_cursor__()
        db = __get_db__()
        cursor.execute(sql)
        db.commit()
    except Exception as x:
        print(x)
        db.rollback()


def modify(message: 'test from python', uid: 1001):
    sql = "update fdt_task_flow set  message = '{}' where uid = {}".format(message, uid)
    print(sql)
    try:

        db = __get_db__()
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
    except Exception as x:
        print(x)
        db.rollback()


class Customer(object):
    def __init__(self, uid, task, module, create_date, message):
        self.uid = uid
        self.name = module
        self.date = create_date
        self.code = message

    def alert(self):
        print("uid={},name={},date={},code={}".format(self.uid, self.name, self.date, self.code))


# modify('python_modify_2020', 1001)
query_one(1001).alert()


def switch(list1):
    for i in list1:
        print(f'i:{i}')
    print('loop end')


# switch(list1=[1, 'xx', 3])


def meta_rows(t):
    print(t[0])
    print(len(t))
    for x in t:
        print(x)


# t = ('骆昊', 38, True, '四川成都')
# meta_rows(t=('骆昊', 38, True, '四川成都'))


def set_utility(set1):
    print(set1)
    print(len(set1))

# set_utility({'x', 'y', 'x'})
