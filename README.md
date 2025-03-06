
================================================================================
FILE: test.py
================================================================================
import configparser
import sys

if len(sys.argv) < 2:
    print("Usage: python check_config.py <config_file>")
    sys.exit(1)

config = configparser.ConfigParser()
config.read(sys.argv[1])

print("Sections in config file:")
for section in config.sections():
    print(f"- {section}")
    for key, value in config[section].items():
        print(f"  • {key} = {value}")

================================================================================
FILE: src\FreeFuzz.py
================================================================================
from utils.skip import need_skip_torch
import configparser
from os.path import join, exists
from os import makedirs
import subprocess
from utils.printer import dump_data
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreeFuzz: a fuzzing frameword for deep learning library")
    parser.add_argument("--conf", type=str, default="demo.conf", help="configuration file")
    args = parser.parse_args()

    config_name = args.conf
    freefuzz_cfg = configparser.ConfigParser()
    freefuzz_cfg.read(join(__file__.replace("FreeFuzz.py", "config"), config_name))

    libs = freefuzz_cfg["general"]["libs"].split(",")
    print("Testing on ", libs)

    mongo_cfg = freefuzz_cfg["mongodb"]
    host = mongo_cfg["host"]
    port = int(mongo_cfg["port"])

    # output configuration - 修正了这里的节名和参数名
    output_cfg = freefuzz_cfg["output"]
    torch_output_dir = output_cfg["torch_output"]
    tf_output_dir = output_cfg["tf_output"]

    # 创建输出目录（如果不存在）
    if not exists(torch_output_dir):
        makedirs(torch_output_dir)
    if not exists(tf_output_dir):
        makedirs(tf_output_dir)

    if "torch" in libs:
        # database configuration
        from classes.database import TorchDatabase

        TorchDatabase.database_config(host, port, mongo_cfg["torch_database"])

        for api_name in TorchDatabase.get_api_list():
            print(api_name)
            if need_skip_torch(api_name):
                continue
            try:
                res = subprocess.run(["python3", "FreeFuzz_api.py", config_name, "torch", api_name], shell=False,
                                     timeout=100)
            except subprocess.TimeoutExpired:
                dump_data(f"{api_name}\n", join(torch_output_dir, "timeout.txt"), "a")
            except Exception as e:
                dump_data(f"{api_name}\n  {e}\n", join(torch_output_dir, "runerror.txt"), "a")
            else:
                if res.returncode != 0:
                    dump_data(f"{api_name}\n", join(torch_output_dir, "runcrash.txt"), "a")
    if "tf" in libs:
        # database configuration
        from classes.database import TFDatabase

        TFDatabase.database_config(host, port, mongo_cfg["tf_database"])

        for api_name in TFDatabase.get_api_list():
            print(api_name)
            try:
                res = subprocess.run(["python3", "FreeFuzz_api.py", config_name, "tf", api_name], shell=False,
                                     timeout=100)
            except subprocess.TimeoutExpired:
                dump_data(f"{api_name}\n", join(tf_output_dir, "timeout.txt"), "a")
            except Exception as e:
                dump_data(f"{api_name}\n  {e}\n", join(tf_output_dir, "runerror.txt"), "a")
            else:
                if res.returncode != 0:
                    dump_data(f"{api_name}\n", join(tf_output_dir, "runcrash.txt"), "a")

    not_test = []
    for l in libs:
        if l not in ["tf", "torch"]: not_test.append(l)
    if len(not_test):
        print(f"WE DO NOT SUPPORT SUCH DL LIBRARY: {not_test}!")

================================================================================
FILE: src\FreeFuzz_api.py
================================================================================
import sys
from constants.enum import OracleType
import configparser
from os.path import join
from utils.converter import str_to_bool


if __name__ == "__main__":
    config_name = sys.argv[1]
    library = sys.argv[2]
    api_name = sys.argv[3]

    freefuzz_cfg = configparser.ConfigParser()
    freefuzz_cfg.read(join(__file__.replace("FreeFuzz_api.py", "config"), config_name))

    # database configuration
    mongo_cfg = freefuzz_cfg["mongodb"]
    host = mongo_cfg["host"]
    port = int(mongo_cfg["port"])

    # oracle configuration
    oracle_cfg = freefuzz_cfg["oracle"]
    crash_oracle = str_to_bool(oracle_cfg["enable_crash"])
    cuda_oracle = str_to_bool(oracle_cfg["enable_cuda"])
    precision_oracle = str_to_bool(oracle_cfg["enable_precision"])

    diff_bound = float(oracle_cfg["float_difference_bound"])
    time_bound = float(oracle_cfg["max_time_bound"])
    time_thresold = float(oracle_cfg["time_thresold"])

    # torch-output configuration
    output_cfg = freefuzz_cfg["torch-output"]
    torch_output_dir = output_cfg["torch-output"]
    tf_output_dir = output_cfg["tf_output"]

    # mutation configuration
    mutation_cfg = freefuzz_cfg["mutation"]
    enable_value = str_to_bool(mutation_cfg["enable_value_mutation"])
    enable_type = str_to_bool(mutation_cfg["enable_type_mutation"])
    enable_db = str_to_bool(mutation_cfg["enable_db_mutation"])
    each_api_run_times = int(mutation_cfg["each_api_run_times"])

    if library.lower() in ["pytorch", "torch"]:
        import torch
        from classes.torch_library import TorchLibrary
        from classes.torch_api import TorchAPI
        from classes.database import TorchDatabase
        from utils.skip import need_skip_torch

        TorchDatabase.database_config(host, port, mongo_cfg["torch_database"])

        if cuda_oracle and not torch.cuda.is_available():
            print("YOUR LOCAL DOES NOT SUPPORT CUDA")
            cuda_oracle = False
        # Pytorch TEST
        MyTorch = TorchLibrary(torch_output_dir, diff_bound, time_bound,
                            time_thresold)
        for _ in range(each_api_run_times):
            api = TorchAPI(api_name)
            api.mutate(enable_value, enable_type, enable_db)
            if crash_oracle:
                MyTorch.test_with_oracle(api, OracleType.CRASH)
            if cuda_oracle:
                MyTorch.test_with_oracle(api, OracleType.CUDA)
            if precision_oracle:
                MyTorch.test_with_oracle(api, OracleType.PRECISION)
    elif library.lower() in ["tensorflow", "tf"]:
        import tensorflow as tf
        from classes.tf_library import TFLibrary
        from classes.tf_api import TFAPI
        from classes.database import TFDatabase
        from utils.skip import need_skip_tf

        TFDatabase.database_config(host, port, mongo_cfg["tf_database"])
        if cuda_oracle and not tf.test.is_gpu_available():
            print("YOUR LOCAL DOES NOT SUPPORT CUDA")
            cuda_oracle = False
        
        MyTF = TFLibrary(tf_output_dir, diff_bound, time_bound,
                            time_thresold)
        print(api_name)
        if need_skip_tf(api_name): pass
        else:
            for _ in range(each_api_run_times):
                api = TFAPI(api_name)
                api.mutate(enable_value, enable_type, enable_db)
                if crash_oracle:
                    MyTF.test_with_oracle(api, OracleType.CRASH)
                if cuda_oracle:
                    MyTF.test_with_oracle(api, OracleType.CUDA)
                if precision_oracle:
                    MyTF.test_with_oracle(api, OracleType.PRECISION)
    else:
        print(f"WE DO NOT SUPPORT SUCH DL LIBRARY: {library}!")


================================================================================
FILE: src\classes\api.py
================================================================================
import inspect
from numpy.random import randint, choice
from classes.argument import ArgType, Argument, OracleType
from utils.probability import *



class API:
    def __init__(self, api_name):
        self.api = api_name

    def mutate(self):
        pass

    def to_code(self) -> str:
        pass

    def to_dict(self) -> dict:
        pass

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        pass

    @staticmethod
    def indent_code(code):
        codes = code.split("\n")
        result = []
        for code in codes:
            if code == "":
                continue
            result.append("  " + code)
        return "\n".join(result) + "\n"



================================================================================
FILE: src\classes\argument.py
================================================================================
from numpy.random import choice, randint
from enum import IntEnum
from utils.probability import *
from constants.enum import OracleType

class ArgType(IntEnum):
    INT = 1
    STR = 2
    FLOAT = 3
    BOOL = 4
    TUPLE = 5
    LIST = 6
    NULL = 7
    TORCH_OBJECT = 8
    TORCH_TENSOR = 9
    TORCH_DTYPE = 10
    TF_TENSOR = 11
    TF_DTYPE = 12
    KERAS_TENSOR = 13
    TF_VARIABLE = 14
    TF_OBJECT = 15
    



class Argument:
    """
    _support_types: all the types that Argument supports.
    NOTICE: The inherent class should call the method of its parent
    when it does not support its type
    """
    _support_types = [
        ArgType.INT, ArgType.STR, ArgType.FLOAT, ArgType.NULL, ArgType.TUPLE,
        ArgType.LIST, ArgType.BOOL
    ]
    _int_values = [-1024, -16, -1, 0, 1, 16, 1024]
    _str_values = [
        "mean", "sum", "max", 'zeros', 'reflect', 'circular', 'replicate'
    ]
    _float_values = [0.0, 1.0, -1.0, 63.0, -63.0, 1024.0, -1024.0, 1e20, -1e20]

    def __init__(self, value, type: ArgType):
        self.value = value
        self.type = type

    def to_code(self, var_name: str) -> str:
        """ArgType.LIST and ArgType.TUPLE should be converted to code in the inherent class"""
        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.BOOL]:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.STR:
            return f"{var_name} = \"{self.value}\"\n"
        elif self.type == ArgType.NULL:
            return f"{var_name} = None\n"
        else:
            assert (0)

    def mutate_value(self) -> None:
        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            self.value = not self.value
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            # self.value is a list now
            for arg in self.value:
                arg.mutate_value()
        elif self.type == ArgType.NULL:
            pass
        else:
            assert (0)

    def mutate_type(self) -> None:
        """The type mutation for NULL should be implemented in the inherent class"""
        if self.type in [
                ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL
        ]:
            types = [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]
            types.remove(self.type)
            self.type = choice(types)
            # change the value
            if self.type == ArgType.INT:
                self.value = self.mutate_int_value(0)
            elif self.type == ArgType.FLOAT:
                self.value = self.mutate_float_value(0.0)
            elif self.type == ArgType.STR:
                self.value = self.mutate_str_value("max")
            elif self.type == ArgType.BOOL:
                self.value = choice([True, False])
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            for arg in self.value:
                arg.mutate_type()
        else:
            # cannot change the type of assert in the general Argument
            assert (0)


    def mutate_int_value(self, value, _min=None, _max=None) -> int:
        if choose_from_list():
            value = choice(Argument._int_values)
        else:
            value += randint(-64, 64)
        # min <= value <= max
        if _min != None:
            value = max(_min, value)
        if _max != None:
            value = min(_max, value)
        return value


    def mutate_str_value(self, value) -> str:
        """You can add more string mutation strategies"""
        if choose_from_list():
            return choice(Argument._str_values)
        else:
            return value


    def mutate_float_value(self, value) -> float:
        if choose_from_list():
            return choice(Argument._float_values)
        else:
            return value + randint(-64, 64) * 1.0


    def initial_value(self, type: ArgType):
        """LIST and TUPLE should be implemented in the inherent class"""
        if type == ArgType.INT:
            return choice(Argument._int_values)
        elif type == ArgType.FLOAT:
            return choice(Argument._float_values)
        elif type == ArgType.STR:
            return choice(Argument._str_values)
        elif type == ArgType.BOOL:
            return choice([True, False])
        elif type == ArgType.NULL:
            return None
        else:
            assert (0)
    
    @staticmethod
    def get_type(x):
        if x is None:
            return ArgType.NULL
        elif isinstance(x, bool):
            return ArgType.BOOL
        elif isinstance(x, int):
            return ArgType.INT
        elif isinstance(x, str):
            return ArgType.STR
        elif isinstance(x, float):
            return ArgType.FLOAT
        elif isinstance(x, tuple):
            return ArgType.TUPLE
        elif isinstance(x, list):
            return ArgType.LIST
        else:
            return None



================================================================================
FILE: src\classes\database.py
================================================================================
import pymongo
from numpy.random import choice
"""
This file is the interfere with database
"""

class Database:
    """Database setting"""
    signature_collection = "signature"
    similarity_collection = "similarity"
    argdef_collection = "api_args"

    def __init__(self) -> None:
        pass

    def database_config(self, host, port, database_name):
        self.DB = pymongo.MongoClient(host=host, port=port)[database_name]

    def index_name(self, api_name, arg_name):
        record = self.DB[self.signature_collection].find_one({"api": api_name})
        if record == None:
            print(f"No such {api_name}")
            return None
        arg_names = record["args"]
        for idx, name in enumerate(arg_names):
            if name == arg_name:
                return f"parameter:{idx}"
        return None

    def select_rand_over_db(self, api_name, arg_name):
        if api_name not in self.DB.list_collection_names():
            return None, False
        arg_names = self.DB[self.signature_collection].find_one({"api": api_name})["args"]
        if arg_name.startswith("parameter:"):
            index = int(arg_name[10:])
            if index >= len(arg_names):
                return None, False
            arg_name = arg_names[index]

        sim_dict = self.DB[self.similarity_collection].find_one({
            "api": api_name,
            "arg": arg_name
        })
        if sim_dict == None:
            return None, False
        APIs = sim_dict["APIs"]
        probs = sim_dict["probs"]
        if len(APIs) == 0:
            return None, False
        target_api = choice(APIs, p=probs)
        # compare the time of 2 operations
        idx_name = self.index_name(target_api, arg_name)
        if idx_name == None:
            return None, False
        select_data = self.DB[target_api].aggregate([{
            "$match": {
                "$or": [{
                    arg_name: {
                        "$exists": True
                    },
                }, {
                    idx_name: {
                        "$exists": True
                    }
                }]
            }
        }, {
            "$sample": {
                "size": 1
            }
        }])
        if not select_data.alive:
            # not found any value in the (target_api, arg_name)
            print(f"ERROR IN SIMILARITY: {target_api}, {api_name}")
            return None, False
        select_data = select_data.next()
        if arg_name in select_data.keys():
            return select_data[arg_name], True
        else:
            return select_data[idx_name], True


    def get_rand_record(self, api_name):
        record = self.DB[api_name].aggregate([{"$sample": {"size": 1}}])
        if not record.alive:
            print(f"NO SUCH API: {api_name}")
            assert(0)
        record = record.next()
        record.pop("_id")
        assert("_id" not in record.keys())
        return record
    
    def get_all_records(self, api_name):
        if api_name not in self.DB.list_collection_names():
            print(f"NO SUCH API: {api_name}")
            return []
        temp = self.DB[api_name].find({}, {"_id": 0})
        records = []
        for t in temp:
            assert("_id" not in t.keys())
            records.append(t)
        return records
    
    def get_signature(self, api_name):
        record = self.DB[self.signature_collection].find_one({"api": api_name}, {"_id": 0})
        if record == None:
            print(f"NO SIGNATURE FOR: {api_name}")
            assert(0)
        return record["args"]

    @staticmethod
    def get_api_list(DB, start_str):
        api_list = []
        for name in DB.list_collection_names():
            if name.startswith(start_str):
                api_list.append(name)
        return api_list

class TorchDB(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "torch.")
        return self.api_list

class TFDB(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "tf.")
        return self.api_list
"""
Database for each library
NOTE:
You must config the database by using `database_config(host, port, name)` before use!!!
Like TFDatabase.database_config("127.0.0.1", 27109, "tftest")
"""
TorchDatabase = TorchDB()
TFDatabase = TFDB()

================================================================================
FILE: src\classes\library.py
================================================================================
from classes.argument import *
from classes.api import *
from os.path import join
import os

class Library:
    def __init__(self, directory) -> None:
        def init_dir(dir_name):
            os.makedirs(join(dir_name, "success"), exist_ok=True)
            os.makedirs(join(dir_name, "potential-bug"), exist_ok=True)
            os.makedirs(join(dir_name, "fail"), exist_ok=True)
            os.makedirs(join(dir_name, "compare-bug"), exist_ok=True)

        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.output = {
            OracleType.CRASH: join(directory, "crash-oracle"),
            OracleType.CUDA: join(directory, "cuda-oracle"),
            OracleType.PRECISION: join(directory, "precision-oracle"),
        }
        for dir_name in self.output.values():
            init_dir(dir_name)
    
    @staticmethod
    def generate_code():
        pass

    @staticmethod
    def write_to_dir(dir, api_name, code):
        api_dir = join(dir, api_name)
        if not os.path.exists(api_dir):
            os.makedirs(api_dir)
        filenames = os.listdir(api_dir)
        max_name = 0
        for name in filenames:
            max_name = max(max_name, int(name.replace(".py", "")))
        new_name = str(max_name + 1)
        with open(join(api_dir, new_name + ".py"), "w") as f:
            f.write(code)


================================================================================
FILE: src\classes\tf_api.py
================================================================================
from functools import WRAPPER_UPDATES
import inspect
import json
import random
from typing import List, Dict
import numpy as np
import tensorflow as tf
from numpy.random import choice, randint

from constants.keys import *
from classes.argument import ArgType, Argument
from classes.api import API
from termcolor import colored

from classes.api import API
from classes.database import TFDatabase

from classes.argument import OracleType
from utils.probability import do_type_mutation, do_select_from_db

class TFArgument(Argument):
    _str_values = ["", "1", "sum", "same", "valid", "zeros"]
    _float_values = [0.0, 1.0, -1.0, 63.0, -63.0]
    _tensor_arg_dtypes = [ArgType.TF_TENSOR, ArgType.KERAS_TENSOR, ArgType.TF_VARIABLE]
    _dtypes = [
        tf.bfloat16, tf.bool, tf.complex128, tf.complex64, tf.double,
        tf.float16, tf.float32, tf.float64, tf.half,
        tf.int16, tf.int32, tf.int64, tf.int8,
        tf.uint8, tf.uint16, tf.uint32, tf.uint64,
    ]
    _support_types = [
        ArgType.TF_TENSOR, ArgType.TF_VARIABLE, ArgType.KERAS_TENSOR,
        ArgType.TF_DTYPE, ArgType.TF_OBJECT
    ]

    def __init__(self, value, type: ArgType, minv=0, maxv=0, shape=None, dtype=None) -> None:
        if isinstance(dtype, str):
            dtype = self.str_to_dtype(dtype)
        shape = self.shape_to_list(shape)

        super().__init__(value, type)
        self.minv = minv
        self.maxv = maxv
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def str_to_dtype(dt: str):
        dt = dt.strip().replace("_ref", "")
        if not dt.startswith("tf."):
            dt = "tf." + dt
        try:
            return eval(dt)
        except:
            return tf.float32

    @staticmethod
    def shape_to_list(shape): 
        if shape is None: return None   
        if not isinstance(shape, list):
            try:
                shape = shape.as_list()
            except:
                shape = list(shape)
            else:
                shape = list(shape)
        shape = [1 if x is None else x for x in shape]
        return shape

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if tf.is_tensor(x):
            if tf.keras.backend.is_keras_tensor(x):
                return ArgType.KERAS_TENSOR
            return ArgType.TF_TENSOR
        elif isinstance(x, tf.DType):
            return ArgType.TF_DTYPE
        
    def mutate_value_random(self) -> None:
        """ Apply random value mutation. """
        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            self.value = self.mutate_bool_value(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for arg in self.value:
                arg.mutate_value_random()
        elif self.type in self._tensor_arg_dtypes:
            self.minv, self.maxv = self.random_tensor_value_range(self.dtype)
        elif self.type == ArgType.TF_DTYPE:
            self.value = TFArgument.mutate_dtype()
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            pass
        else:
            raise ValueError(self.type)
            assert (0)

    def if_mutate_shape(self):
        return random.random() < 0.3

    def if_mutate_shape_value(self):
        return random.random() < 0.3

    def if_expand_dim(self):
        return random.random() < 0.3

    def if_squeeze(self):
        return random.random() < 0.3

    def mutate_shape(self, old_shape):
        new_shape = old_shape

        # Change rank
        if self.if_expand_dim():
            new_shape.append(1)
        elif len(new_shape) > 0 and self.if_squeeze():
            new_shape.pop()
        # Change value
        for i in range(len(new_shape)):
            if self.if_mutate_shape_value():
                new_shape[i] = self.mutate_int_value(new_shape[i], minv=0)
               
        return new_shape

    def generate_value_random(self) -> None:

        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(0)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value("")
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(0.)
        elif self.type == ArgType.BOOL:
            self.value = self.mutate_bool_value(True)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            self.value = [TFArgument(1, ArgType.INT), TFArgument(1, ArgType.INT)]
        elif self.type in self._tensor_arg_dtypes:
            shape = [randint(1, 3), randint(1, 3)]
            dtype = choice([tf.int32, tf.float32, tf.float64])
            self.shape, self.dtype = shape, dtype
            self.value, self.minv, self.maxv = None, 0, 1
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TF_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            assert (0)

    def mutate_type(self) -> None:
        def if_mutate_primitive():
            return random.random() < 0.1

        def if_mutate_null():
            return random.random() < 0.1

        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]:
            if not if_mutate_primitive(): return False
            # change the type
            types = [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]
            types.remove(self.type)
            self.type = choice(types)
            # change the value
            if self.type == ArgType.INT:
                self.value = self.mutate_int_value(0)
            elif self.type == ArgType.FLOAT:
                self.value = self.mutate_float_value(0.0)
            elif self.type == ArgType.STR:
                self.value = self.mutate_str_value("")
            elif self.type == ArgType.BOOL:
                self.value = choice([True, False])
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            if random.random() < 0.01: 
                self.value = [] # with a probability return an empty list
            for arg in self.value:
                arg.mutate_type()
        elif self.type == ArgType.TF_TENSOR:
            dtype = choice(self._dtypes)
            shape = self.shape
            if self.if_mutate_shape():
                shape = self.mutate_shape(shape)
            self.shape, self.dtype = shape, dtype
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            if not if_mutate_null():
                return False
            new_type = choice(self._support_types + super()._support_types)
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    TFArgument(2, ArgType.INT),
                    TFArgument(3, ArgType.INT)
                ]
            elif new_type == ArgType.TF_TENSOR:
                self.shape = [2, 2]
                self.dtype = tf.float32

            if new_type != ArgType.NULL:
                try:
                    self.type = new_type
                    self.generate_value_random()
                except:
                    pass
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(TFArgument._dtypes)
        return True

    @staticmethod
    def if_mutate_int_random():
        return random.random() < 0.2

    @staticmethod
    def if_mutate_str_random():
        return random.random() < 0.1

    @staticmethod
    def if_mutate_float_random():
        return random.random() < 0.2

    
    def mutate_bool_value(self, value) -> bool:
        return choice([True, False])

    def mutate_int_value(self, value, minv=None, maxv=None) -> int:
        if TFArgument.if_mutate_int_random():
            value = choice(self._int_values)
        else:
            value += randint(-2, 2)
        if minv is not None:
            value = max(minv, value)
        if maxv is not None:
            value = min(maxv, value)
        return value
    
    def mutate_str_value(self, value) -> str:
        if TFArgument.if_mutate_str_random():
            return choice(self._str_values)
        return value

    @staticmethod
    def mutate_dtype() -> tf.dtypes.DType:
        return choice(TFArgument._dtypes)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [tf.int16, tf.int32, tf.int64]:
            return tf.int8
        elif dtype in [tf.float32, tf.float64]:
            return tf.float16
        elif dtype in [ tf.complex128]:
            return tf.complex64
        return dtype

    @staticmethod
    def random_tensor_value_range(dtype):
        assert isinstance(dtype, tf.dtypes.DType)
        minv = 0
        maxv = 1
        if dtype.is_floating or dtype.is_complex or dtype == tf.string or dtype == tf.bool:
            pass
        elif "int64" in dtype.name or "int32" in dtype.name or "int16" in dtype.name:
            minv = 0 if "uint" in dtype.name else - (1 << 8)
            maxv = (1 << 8)
        else:
            try:
                minv = dtype.min
                maxv = dtype.max
            except Exception as e:
                minv, maxv = 0, 1
        return minv, maxv

    def to_code_tensor(self, var_name, low_precision=False):
        dtype = self.dtype
        if low_precision:
            dtype = self.low_precision_dtype(dtype)
        shape = self.shape
        if dtype is None:
            assert (0)
        code = ""
        var_tensor_name = f"{var_name}_tensor"
        if dtype.is_floating:
            code += "%s = tf.random.uniform(%s, dtype=tf.%s)\n" % (var_tensor_name, shape, dtype.name)
        elif dtype.is_complex:
            ftype = "float64" if dtype == tf.complex128 else "float32"
            code += "%s = tf.complex(tf.random.uniform(%s, dtype=tf.%s)," \
                    "tf.random.uniform(%s, dtype=tf.%s))\n" % (var_tensor_name, shape, ftype, shape, ftype)
        elif dtype == tf.bool:
            code += "%s = tf.cast(tf.random.uniform(" \
                   "%s, minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)\n" % (var_tensor_name, shape)
        elif dtype == tf.string:
            code += "%s = tf.convert_to_tensor(np.ones(%s, dtype=str))\n" % (var_tensor_name, shape)
        elif dtype in [tf.int32, tf.int64]:
            code += "%s = tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.%s)\n" \
                % (var_tensor_name, shape, self.minv, self.maxv + 1, dtype.name)
        else:
            code += "%s = tf.saturate_cast(" \
                "tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.int64), " \
                "dtype=tf.%s)\n" % (var_tensor_name, shape, self.minv, self.maxv + 1, dtype.name)
        code += f"{var_name} = tf.identity({var_tensor_name})\n"
        return code

    def to_code_keras_tensor(self, var_name, low_precision=False):
        return self.to_code_tensor(var_name, low_precision=low_precision)

    def to_code(self, var_name, low_precision=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}", low_precision)
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, self.value.name)
        elif self.type in self._tensor_arg_dtypes:
            # Did not consider cloning for in-place operation here.
            code = ""
            if self.type == ArgType.TF_TENSOR:
                code = self.to_code_tensor(var_name, low_precision=low_precision)
            elif self.type == ArgType.TF_VARIABLE:
                code = self.to_code_tensor(var_name, low_precision=low_precision)
                code += "%s = tf.Variable(%s)\n" % (var_name, var_name)
            elif self.type == ArgType.KERAS_TENSOR:
                code = self.to_code_keras_tensor(var_name, low_precision=low_precision)
            return code
        return super().to_code(var_name)


    def to_diff_code(self, var_name, low_precision=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", low_precision)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, self.value.name)
        elif self.type in self._tensor_arg_dtypes:
            code = f"{var_name} = tf.identity({var_name}_tensor)\n"
            if not low_precision:
                code += f"{var_name} = tf.cast({var_name}, tf.{self.dtype.name})\n"
            if self.type == ArgType.TF_VARIABLE:
                code += "%s = tf.Variable(%s)\n" % (var_name, var_name)
            return code
        return ""

    def mutate_value(self):
        self.mutate_value_random()

    @staticmethod
    def generate_arg_from_signature(signature):
        if isinstance(signature, bool):
            return TFArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return TFArgument(signature, ArgType.INT)
        if isinstance(signature, float):
            return TFArgument(signature, ArgType.FLOAT)
        if isinstance(signature, str):
            return TFArgument(signature, ArgType.STR)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TFArgument.generate_arg_from_signature(elem))
            return TFArgument(value, ArgType.LIST)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TFArgument.generate_arg_from_signature(elem))
            return TFArgument(value, ArgType.TUPLE)

        if (not isinstance(signature, dict)):
            return TFArgument(None, ArgType.NULL)

        if "type" not in signature and "Label" not in signature:
            return TFArgument(None, ArgType.NULL)

        label = signature["type"] if "type" in signature else signature["Label"]

        if label == "tf_object":
            if "class_name" not in signature:
                return TFArgument(None, ArgType.TF_OBJECT)

            if signature["class_name"] == "tensorflow.python.keras.engine.keras_tensor.KerasTensor" or \
                signature["class_name"] == "tensorflow.python.ops.variables.RefVariable":
                dtype = signature["dtype"]
                shape = signature["shape"]
                dtype = TFArgument.str_to_dtype(dtype)
                minv, maxv = TFArgument.random_tensor_value_range(dtype)
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)
            if signature["class_name"] == "tensorflow.python.framework.dtypes.DType":
                name = signature["to_str"].replace("<dtype: '", "").replace("'>", "")
                value = eval("tf." + name)
                return TFArgument(value, ArgType.TF_DTYPE)
            try:
                value = eval(signature.class_name)
            except:
                value = None
            return TFArgument(value, ArgType.TF_OBJECT)
        if label == "raw":
            try:
                value = json.loads(signature['value'])
            except:
                value = signature['value']
                pass
            if isinstance(value, int):
                return TFArgument(value, ArgType.INT)
            if isinstance(value, str):
                return TFArgument(value, ArgType.STR)
            if isinstance(value, float):
                return TFArgument(value, ArgType.FLOAT)
            if isinstance(value, tuple):
                tuple_value = []
                for elem in value:
                    tuple_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(tuple_value, ArgType.TUPLE)
            if isinstance(value, list):
                list_value = []
                for elem in value:
                    list_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(list_value, ArgType.LIST)

        if label == "tuple":
            try:
                value = json.loads(signature['value'])
                tuple_value = []
                for elem in value:
                    tuple_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(tuple_value, ArgType.TUPLE)
            except:
                raise ValueError("Wrong signature " + str(signature))
        if label == "list":
            try:
                try:
                    value = json.loads(signature['value'])
                except:
                    value = signature['value']
                list_value = []
                for elem in value:
                    list_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(list_value, ArgType.LIST)
            except:
                raise ValueError("Wrong signature " + str(signature))
        if label in ["tensor", "KerasTensor", "variable", "nparray"]:
            if not ('shape' in signature.keys()
                    and 'dtype' in signature.keys()):
                raise Exception('Wrong signature {0}'.format(signature))
            shape = signature['shape']
            dtype = signature["dtype"]
            dtype = TFArgument.str_to_dtype(dtype)

            if isinstance(shape, (list, tuple)):
                minv, maxv = TFArgument.random_tensor_value_range(dtype)
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)
            else:
                minv, maxv = 0, 1
                shape = [1, ]  
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)

        return TFArgument(None, ArgType.NULL)

class TFAPI(API):
    def __init__(self, api_name, record=None) -> None:
        super().__init__(api_name)
        self.record = TFDatabase.get_rand_record(api_name) if record is None else record
        self.args = TFAPI.generate_args_from_record(self.record)
        self.is_class = inspect.isclass(eval(self.api))

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = TFDatabase.select_rand_over_db(self.api, arg_name)
                if success:
                    new_arg = TFArgument.generate_arg_from_signature(new_arg)
                    self.args[arg_name] = new_arg
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def to_code_oracle(self,
                prefix="arg", oracle=OracleType.CRASH) -> str:
        
        if oracle == OracleType.CRASH:
            code = self.to_code(prefix=prefix, res_name=RESULT_KEY)
            return self.wrap_try(code, ERROR_KEY)
        elif oracle == OracleType.CUDA:
            cpu_code = self.to_code(prefix=prefix, res_name=RES_CPU_KEY, 
                use_try=True, err_name=ERR_CPU_KEY, wrap_device=True, device_name="CPU")
            gpu_code = self.to_diff_code(prefix=prefix, res_name=RES_GPU_KEY,
                use_try=True, err_name=ERR_GPU_KEY, wrap_device=True, device_name="GPU:0")
            
            code = cpu_code + gpu_code
            return self.wrap_try(code, ERROR_KEY)
        elif oracle == OracleType.PRECISION:
            low_code = self.to_code(prefix=prefix, res_name=RES_LOW_KEY, low_precision=True,
                use_try=True, err_name=ERR_LOW_KEY, time_it=True, time_var=TIME_LOW_KEY)
            high_code = self.to_diff_code(prefix=prefix, res_name=RES_HIGH_KEY,
                use_try=True, err_name=ERR_HIGH_KEY, time_it=True, time_var=TIME_HIGH_KEY)
            code = low_code + high_code
            return self.wrap_try(code, ERROR_KEY)
        return ''

    @staticmethod
    def generate_args_from_record(record: dict):

        def generate_args_from_signatures(signatures):
            if isinstance(signatures, dict):
                if signatures['Label'] == 'list':
                    s = signatures['value']
                    if isinstance(s, list):
                        signatures = s
            args = []
            if signatures == None:
                return args
            for signature in signatures:
                x = TFArgument.generate_arg_from_signature(signature)
                args.append(x)
            return args

        args = {}
        for key in record.keys():
            if key == "input_signature":
                value = generate_args_from_signatures(record[key])
                args[key] = TFArgument(value, ArgType.LIST)
            elif key != "output_signature":
                args[key] = TFArgument.generate_arg_from_signature(record[key])
        return args

    def _to_arg_code(self, prefix="arg", low_precision=False):
        args = []
        kwargs = {}
        for key in self.args.keys():
            if "parameter:" in key:
                args.append(self.args[key])
            elif key != "output_signature" and key != "input_signature":
                kwargs[key] = self.args[key]

        arg_code = ""
        arg_str = ""
        index = 0
        for arg in args:
            arg_code += arg.to_code(f"{prefix}_{index}", low_precision=low_precision)
            arg_str += f"{prefix}_{index},"
            index += 1
        for key, arg in kwargs.items():
            arg_code += arg.to_code(key, low_precision=low_precision)
            arg_str += "%s=%s," % (key, key)
        return arg_code, arg_str
        
    def _to_diff_arg_code(self, prefix="arg", low_precision=False):
        args = []
        kwargs = {}
        for key in self.args.keys():
            if "parameter:" in key:
                args.append(self.args[key])
            elif key != "output_signature" and key != "input_signature":
                kwargs[key] = self.args[key]

        arg_code = ""
        arg_str = ""
        index = 0
        for arg in args:
            arg_code += arg.to_diff_code(f"{prefix}_{index}", low_precision=low_precision)
            arg_str += f"{prefix}_{index},"
            index += 1
        for key, arg in kwargs.items():
            arg_code += arg.to_diff_code(key, low_precision=low_precision)
            arg_str += "%s=%s," % (key, key)
        return arg_code, arg_str

    def to_code(self, prefix="arg", res_name="", low_precision=False, **kwargs) -> str:
        
        inputs = None
        input_name = ''
        if "input_signature" in self.args:
            inputs = self.args["input_signature"]
        if inputs:
            input_name = f"{prefix}_input"

        arg_code, arg_str = self._to_arg_code(prefix=prefix, low_precision=low_precision)
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            arg_code += f"{cls_name} = {self.api}({arg_str})\n"
            if inputs:
                arg_code += inputs.to_code(input_name, low_precision=low_precision)
                res_code += f"{RES_KEY}[\"{res_name}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res_name}\"] = {self.api}({arg_str})\n"
        invocation = self._to_invocation_code(arg_code, res_code, **kwargs)
        return invocation

    def to_diff_code(self, prefix="arg", res_name="", low_precision=False, **kwargs) -> str:
        
        inputs = None
        input_name = ''
        if "input_signature" in self.args:
            inputs = self.args["input_signature"]
        if inputs:
            input_name = f"{prefix}_input"

        arg_code, arg_str = self._to_diff_arg_code(prefix=prefix, low_precision=low_precision)
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            res_code = f""
            if inputs:
                arg_code += inputs.to_diff_code(input_name, low_precision=low_precision)
                res_code += f"{RES_KEY}[\"{res_name}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res_name}\"] = {self.api}({arg_str})\n"
        
        invocation = self._to_invocation_code(arg_code, res_code, **kwargs)
        return invocation

    def _to_res_code(self, res_name, arg_str, input_name=None, prefix="arg"):
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            if input_name:
                res_code += f"{RES_KEY}[\"{res_name}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res_name}\"] = {self.api}({arg_str})\n"
        return res_code

    def _to_invocation_code(self, arg_code, res_code, use_try=False, err_name="", 
        wrap_device=False, device_name="", time_it=False, time_var="", **kwargs) -> str:
        if time_it:
            res_code = res_code + self.wrap_time(res_code, time_var)
        code = arg_code + res_code
        inv_code = code
        if wrap_device:
            inv_code = self.wrap_device(inv_code, device=device_name)
        if use_try:
            inv_code = self.wrap_try(inv_code, error_var=err_name)
        return inv_code

    @staticmethod
    def wrap_try(code:str, error_var) -> str:
        wrapped_code = "try:\n"
        if code.strip() == "":
            code = "pass"
        wrapped_code += API.indent_code(code)
        wrapped_code += f"except Exception as e:\n  {RES_KEY}[\"{error_var}\"] = \"Error:\"+str(e)\n"
        return wrapped_code

    @staticmethod
    def wrap_device(code:str, device) -> str:
        device_code = f"with tf.device('/{device}'):\n" + API.indent_code(code)
        return device_code

    @staticmethod
    def wrap_time(code:str, time_var) -> str:
        wrapped_code = "t_start = time.time()\n"
        wrapped_code += code
        wrapped_code += "t_end = time.time()\n"
        wrapped_code += f"{RES_KEY}[\"{time_var}\"] = t_end - t_start\n"
        return wrapped_code


        
def test_tf_arg():
    arg = TFArgument(None, ArgType.TF_TENSOR, shape=[2, 2], dtype=tf.int64)
    arg.mutate_value()
    print(arg.to_code("var"))
    print(arg.to_code("var", True))

def test_tf_api():
    api_name = "tf.keras.layers.Conv2D"
    record = TFDatabase.get_rand_record(api_name)
    api = TFAPI(api_name, record)
    api.mutate()
    print(api.to_code_oracle(oracle=OracleType.CRASH))
    print(api.to_code_oracle(oracle=OracleType.CUDA))
    print(api.to_code_oracle(oracle=OracleType.PRECISION))

if __name__ == '__main__':
    # test_tf_arg()
    test_tf_api()


================================================================================
FILE: src\classes\tf_library.py
================================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os.path import join
import tensorflow as tf
import time
import numpy as np

from classes.argument import Argument, ArgType
from classes.tf_api import TFAPI, TFArgument
from classes.library import Library
from classes.database import TFDatabase
from constants.enum import OracleType
from constants.keys import ERR_CPU_KEY, ERR_GPU_KEY, ERR_HIGH_KEY, ERR_LOW_KEY, ERROR_KEY, RES_CPU_KEY, RES_GPU_KEY, TIME_HIGH_KEY, TIME_LOW_KEY

class TFLibrary(Library):
    def __init__(self, output_dir, diff_bound=1e-5, time_bound=10, time_thresold=1e-3) -> None:
        super().__init__(output_dir)
        self.diff_bound = diff_bound
        self.time_bound = time_bound
        self.time_thresold = time_thresold
    
    def test_with_oracle(self, api: TFAPI, oracle: OracleType):
        if oracle == OracleType.CRASH:
            # We need call another process to catch the crash error
            code = "import tensorflow as tf\n"
            code += self.generate_code(api, oracle)

            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(code)
            results, error = self.run_code(code)
            if error == None:
                self.write_to_dir(join(self.output[oracle], "success"), api.api, code)
            else:
                self.write_to_dir(join(self.output[oracle], "fail"), api.api, code)
        elif oracle == OracleType.CUDA:
            code = "import tensorflow as tf\n"
            code += self.generate_code(api, oracle)

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)
            err_cpu = results[ERR_CPU_KEY]
            err_gpu = results[ERR_GPU_KEY]
            write_dir = ""
            if error is None:
                if (err_cpu is None) != (err_gpu is None):
                    write_dir = join(self.output[oracle], "potential-bug")
                elif err_cpu == None:
                    res_cpu = results[RES_CPU_KEY]
                    res_gpu = results[RES_GPU_KEY]
                    if self.is_equal(res_cpu, res_gpu):
                        write_dir = join(self.output[oracle], "success")
                    else:
                        write_dir = join(self.output[oracle], "potential-bug")
                elif "SystemError" in err_cpu or "SystemError" in err_gpu:
                    write_dir = join(self.output[oracle], "potential-bug")
                else:
                    write_dir = join(self.output[oracle], "success")
            elif "SystemError" in error:
                write_dir = join(self.output[oracle], "potential-bug")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)
        elif oracle == OracleType.PRECISION:
            code = "import tensorflow as tf\n"
            code += "import time\n"
            code += self.generate_code(api, oracle)

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)
            err_high = results[ERR_HIGH_KEY]
            err_low = results[ERR_LOW_KEY]
            write_dir = ""
            if error is None:
                if (err_high is None) != (err_low is None):
                    write_dir = join(self.output[oracle], "potential-bug")
                elif err_high == None:
                    time_high = results[TIME_HIGH_KEY]
                    time_low = results[TIME_LOW_KEY]
                    if time_low >= self.time_bound * time_high and time_high >= self.time_thresold:
                        write_dir = join(self.output[oracle], "potential-bug")
                    else:
                        write_dir = join(self.output[oracle], "success")
                elif "SystemError" in err_high or "SystemError" in err_low:
                    write_dir = join(self.output[oracle], "potential-bug")
                else:
                    write_dir = join(self.output[oracle], "success")
            elif "SystemError" in error:
                write_dir = join(self.output[oracle], "potential-bug")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)

    @staticmethod
    def generate_code(api: TFAPI, oracle: OracleType) -> str:
        code = ""
        if oracle == OracleType.CRASH:
            code += api.to_code_oracle(oracle=oracle)
            return code
        elif oracle == OracleType.CUDA:
            code += api.to_code_oracle(oracle=oracle)
            return code
        elif oracle == OracleType.PRECISION:
            code += api.to_code_oracle(oracle=oracle)
            return code
        else:
            assert(0)
    
    @staticmethod
    def run_code(code):
        results = dict()
        results[ERR_CPU_KEY] = None
        results[ERR_GPU_KEY] = None
        results[ERR_HIGH_KEY] = None
        results[ERR_LOW_KEY] = None
        
        exec(code)
        error = results[ERROR_KEY] if ERROR_KEY in results else None
        return results, error
    
    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if isinstance(x, tf.Tensor):
            return ArgType.TF_TENSOR
        elif isinstance(x, tf.DType):
            return ArgType.TF_DTYPE
        else:
            return ArgType.TF_OBJECT

    
    @staticmethod
    def _eval_k(x):
        return tf.convert_to_tensor(x).numpy()

    @staticmethod
    def get_tensor_value(t):
        if isinstance(t, tf.SparseTensor):
            return tf.sparse.to_dense(t).numpy()
        else:
            return t.numpy()
            
    @staticmethod
    def is_equal(x, y):
        x_type = TFArgument.get_type(x)
        y_type = TFArgument.get_type(y)
        if x_type != y_type:
            return False
        if x_type == ArgType.KERAS_TENSOR:
            return tf.math.equal(x, y)
        if x_type == ArgType.TF_TENSOR:
            try:
                if isinstance(x, tf.RaggedTensor) != isinstance(y, tf.RaggedTensor):
                    return False
                if isinstance(x, tf.RaggedTensor):
                    s = tf.math.equal(x, y)
                    return s.flat_values.numpy().all()
                np_x = TFLibrary.get_tensor_value(x)
                np_y = TFLibrary.get_tensor_value(y)
                if x.dtype.is_floating:
                    return tf.experimental.numpy.allclose(np_x, np_y, rtol=1e-3, atol=1e-4)
                elif x.dtype.is_integer:
                    return np.equal(np_x, np_y).all()
            except:
                raise ValueError(f"Comparison between {type(x)} is not supported now.")
            return True
        elif x_type == ArgType.FLOAT:
            return abs(x - y) < 1e-5
        elif x_type in [ArgType.LIST, ArgType.TUPLE]:
            if len(x) != len(y):
                return False
            for i in range(len(x)):
                if TFLibrary.is_equal(x[i], y[i]) == False:
                    return False
            return True
        
        else:
            try:
                flag = x == y
            except:
                return True

            if isinstance(flag, np.ndarray):
                flag = flag.all()
            try:
                if flag:
                    pass
            except:
                flag = True
            return flag
    


================================================================================
FILE: src\classes\torch_api.py
================================================================================
import torch
from classes.argument import *
from classes.api import *
from classes.database import TorchDatabase
from os.path import join

class TorchArgument(Argument):
    _supported_types = [
        ArgType.TORCH_DTYPE, ArgType.TORCH_OBJECT, ArgType.TORCH_TENSOR
    ]
    _dtypes = [
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        torch.float16, torch.float32, torch.float64, torch.bfloat16,
        torch.complex64, torch.complex128, torch.bool
    ]
    _memory_format = [
        torch.contiguous_format, torch.channels_last, torch.preserve_format
    ]

    def __init__(self,
                 value,
                 type: ArgType,
                 shape=None,
                 dtype=None,
                 max_value=1,
                 min_value=0):
        super().__init__(value, type)
        self.shape = shape
        self.dtype = dtype
        self.max_value = max_value
        self.min_value = min_value

    def to_code(self, var_name, low_precision=False, is_cuda=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}", low_precision,
                                              is_cuda)
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TORCH_TENSOR:
            dtype = self.dtype
            max_value = self.max_value
            min_value = self.min_value
            if low_precision:
                dtype = self.low_precision_dtype(dtype)
                max_value, min_value = self.random_tensor_value(dtype)
            suffix = ""
            if is_cuda:
                suffix = ".cuda()"
            if dtype.is_floating_point:
                code = f"{var_name}_tensor = torch.rand({self.shape}, dtype={dtype})\n"
            elif dtype.is_complex:
                code = f"{var_name}_tensor = torch.rand({self.shape}, dtype={dtype})\n"
            elif dtype == torch.bool:
                code = f"{var_name}_tensor = torch.randint(0,2,{self.shape}, dtype={dtype})\n"
            else:
                code = f"{var_name}_tensor = torch.randint({min_value},{max_value},{self.shape}, dtype={dtype})\n"
            code += f"{var_name} = {var_name}_tensor.clone(){suffix}\n"
            return code
        elif self.type == ArgType.TORCH_OBJECT:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.TORCH_DTYPE:
            return f"{var_name} = {self.value}\n"
        return super().to_code(var_name)

    def to_diff_code(self, var_name, oracle):
        """differential testing with oracle"""
        code = ""
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", oracle)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
        elif self.type == ArgType.TORCH_TENSOR:
            if oracle == OracleType.CUDA:
                code += f"{var_name} = {var_name}_tensor.clone().cuda()\n"
            elif oracle == OracleType.PRECISION:
                code += f"{var_name} = {var_name}_tensor.clone().type({self.dtype})\n"
        return code

    def mutate_value(self) -> None:
        if self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TORCH_TENSOR:
            self.max_value, self.min_value = self.random_tensor_value(
                self.dtype)
        elif self.type in super()._support_types:
            super().mutate_value()
        else:
            print(self.type, self.value)
            assert (0)

    def mutate_type(self) -> None:
        if self.type == ArgType.NULL:
            # choose from all types
            new_type = choice(self._support_types + super()._support_types)
            self.type = new_type
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    TorchArgument(2, ArgType.INT),
                    TorchArgument(3, ArgType.INT)
                ]
            elif new_type == ArgType.TORCH_TENSOR:
                self.shape = [2, 2]
                self.dtype = torch.float32
            elif new_type == ArgType.TORCH_DTYPE:
                self.value = choice(self._dtypes)
            elif new_type == ArgType.TORCH_OBJECT:
                self.value = choice(self._memory_format)
            else:
                self.value = super().initial_value(new_type)
        elif self.type == ArgType.TORCH_TENSOR:
            new_size = list(self.shape)
            # change the dimension of tensor
            if change_tensor_dimension():
                if add_tensor_dimension():
                    new_size.append(1)
                elif len(new_size) > 0:
                    new_size.pop()
            # change the shape
            for i in range(len(new_size)):
                if change_tensor_shape():
                    new_size[i] = self.mutate_int_value(new_size[i], _min=0)
            self.shape = new_size
            # change dtype
            if change_tensor_dtype():
                self.dtype = choice(self._dtypes)
                self.max_value, self.min_value = self.random_tensor_value(self.dtype)
        elif self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type in super()._support_types:
            super().mutate_type()
        else:
            print(self.type, self.value)
            assert (0)

    @staticmethod
    def random_tensor_value(dtype):
        max_value = 1
        min_value = 0
        if dtype == torch.bool:
            max_value = 2
            min_value = 0
        elif dtype == torch.uint8:
            max_value = 1 << randint(0, 9)
            min_value = 0
        elif dtype == torch.int8:
            max_value = 1 << randint(0, 8)
            min_value = -1 << randint(0, 8)
        elif dtype == torch.int16:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        else:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        return max_value, min_value

    @staticmethod
    def generate_arg_from_signature(signature):
        """Generate a Torch argument from the signature"""
        if signature == "torchTensor":
            return TorchArgument(None,
                                 ArgType.TORCH_TENSOR,
                                 shape=[2, 2],
                                 dtype=torch.float32)
        if signature == "torchdtype":
            return TorchArgument(choice(TorchArgument._dtypes),
                                 ArgType.TORCH_DTYPE)
        if isinstance(signature, str) and signature == "torchdevice":
            value = torch.device("cpu")
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torchmemory_format":
            value = choice(TorchArgument._memory_format)
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torch.strided":
            return TorchArgument("torch.strided", ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature.startswith("torch."):
            value = eval(signature)
            if isinstance(value, torch.dtype):
                return TorchArgument(value, ArgType.TORCH_DTYPE)
            elif isinstance(value, torch.memory_format):
                return TorchArgument(value, ArgType.TORCH_OBJECT)
            print(signature)
            assert(0)
        if isinstance(signature, bool):
            return TorchArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return TorchArgument(signature, ArgType.INT)
        if isinstance(signature, str):
            return TorchArgument(signature, ArgType.STR)
        if isinstance(signature, float):
            return TorchArgument(signature, ArgType.FLOAT)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.TUPLE)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.LIST)
        # signature is a dictionary
        if isinstance(signature, dict):
            if not ('shape' in signature.keys()
                    and 'dtype' in signature.keys()):
                raise Exception('Wrong signature {0}'.format(signature))
            shape = signature['shape']
            dtype = signature['dtype']
            # signature is a ndarray or tensor.
            if isinstance(shape, (list, tuple)):
                if not dtype.startswith("torch."):
                    dtype = f"torch.{dtype}"
                dtype = eval(dtype)
                max_value, min_value = TorchArgument.random_tensor_value(dtype)
                return TorchArgument(None,
                                     ArgType.TORCH_TENSOR,
                                     shape,
                                     dtype=dtype,
                                     max_value=max_value,
                                     min_value=min_value)
            else:
                return TorchArgument(None,
                                     ArgType.TORCH_TENSOR,
                                     shape=[2, 2],
                                     dtype=torch.float32)
        return TorchArgument(None, ArgType.NULL)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [torch.int16, torch.int32, torch.int64]:
            return torch.int8
        elif dtype in [torch.float32, torch.float64]:
            return torch.float16
        elif dtype in [torch.complex64, torch.complex128]:
            return torch.complex32
        return dtype

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if isinstance(x, torch.Tensor):
            return ArgType.TORCH_TENSOR
        elif isinstance(x, torch.dtype):
            return ArgType.TORCH_DTYPE
        else:
            return ArgType.TORCH_OBJECT


class TorchAPI(API):
    def __init__(self, api_name, record=None):
        super().__init__(api_name)
        if record == None:
            record = TorchDatabase.get_rand_record(self.api)
        self.args = self.generate_args_from_record(record)
        self.is_class = inspect.isclass(eval(self.api))

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = TorchDatabase.select_rand_over_db(
                    self.api, arg_name)
                if success:
                    new_arg = TorchArgument.generate_arg_from_signature(
                        new_arg)
                    self.args[arg_name] = new_arg
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def to_code(self,
                prefix="arg",
                res="res",
                is_cuda=False,
                use_try=False,
                error_res=None,
                low_precision=False) -> str:
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_code(arg_name,
                                low_precision=low_precision,
                                is_cuda=is_cuda)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if is_cuda:
                code += f"{prefix}_class = {self.api}({arg_str}).cuda()\n"
            else:
                code += f"{prefix}_class = {self.api}({arg_str})\n"

            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_code(
                    arg_name, low_precision=low_precision, is_cuda=is_cuda)
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(res, error_res, res_code, use_try,
                                           low_precision)

    def to_diff_code(self,
                     oracle: OracleType,
                     prefix="arg",
                     res="res",
                     *,
                     error_res=None,
                     use_try=False) -> str:
        """Generate code for the oracle"""
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_diff_code(arg_name, oracle)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if oracle == OracleType.CUDA:
                code = f"{prefix}_class = {prefix}_class.cuda()\n"
            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_diff_code(arg_name, oracle)
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(res, error_res, res_code, use_try,
                                           oracle == OracleType.PRECISION)

    @staticmethod
    def invocation_code(res, error_res, res_code, use_try, low_precision):
        code = ""
        if use_try:
            # specified with run_and_check function in relation_tools.py
            if error_res == None:
                error_res = res
            temp_code = "try:\n"
            temp_code += API.indent_code(res_code)
            temp_code += f"except Exception as e:\n  {error_res} = \"ERROR:\"+str(e)\n"
            res_code = temp_code

        if low_precision:
            code += "start = time.time()\n"
            code += res_code
            code += f"{res} = time.time() - start\n"
        else:
            code += res_code
        return code

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        args = {}
        for key in record.keys():
            if key != "output_signature":
                args[key] = TorchArgument.generate_arg_from_signature(
                    record[key])
        return args

================================================================================
FILE: src\classes\torch_library.py
================================================================================
from classes.torch_api import *
from classes.library import Library
from classes.argument import *
from classes.api import *
from os.path import join
import os
from constants.keys import *

class TorchLibrary(Library):
    def __init__(self, output_dir, diff_bound=1e-5, time_bound=10, time_thresold=1e-3) -> None:
        super().__init__(output_dir)
        self.diff_bound = diff_bound
        self.time_bound = time_bound
        self.time_thresold = time_thresold
    
    def test_with_oracle(self, api: TorchAPI, oracle: OracleType):
        if oracle == OracleType.CRASH:
            # We need call another process to catch the crash error
            code = "import torch\n"
            code += self.generate_code(api, oracle)
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(code)
            _, error = self.run_code(code)
            if error == None:
                self.write_to_dir(join(self.output[oracle], "success"), api.api, code)
            elif self.is_crash_msg(error):
                self.write_to_dir(join(self.output[oracle], "potential-bug"), api.api, code)
            else:
                self.write_to_dir(join(self.output[oracle], "fail"), api.api, code)
        elif oracle == OracleType.CUDA:
            code = "import torch\n"
            code += api.to_code(res=f"{RES_KEY}[\"{RES_CPU_KEY}\"]", use_try=True, error_res=f"{RES_KEY}[\"{ERR_CPU_KEY}\"]")
            code += api.to_diff_code(oracle, res=f"{RES_KEY}[\"{RES_GPU_KEY}\"]", use_try=True, error_res=f"{RES_KEY}[\"{ERR_GPU_KEY}\"]")

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)

            write_dir = ""
            if error == None:
                # first check the correctness
                if results[ERR_CPU_KEY] == None and results[ERR_GPU_KEY] == None:
                    try:
                        is_equal = self.is_equal(results[RES_CPU_KEY], results[RES_GPU_KEY], self.diff_bound)
                    except Exception:
                        write_dir = join(self.output[oracle], "compare-bug")
                    else:
                        if is_equal:
                            write_dir = join(self.output[oracle], "success")
                        else:
                            write_dir = join(self.output[oracle], "potential-bug")
                elif self.is_crash_msg(results[ERR_CPU_KEY]) or self.is_crash_msg(results[ERR_GPU_KEY]):
                    write_dir = join(self.output[oracle], "potential-bug")
                elif results[ERR_CPU_KEY] and results[ERR_GPU_KEY]:
                    write_dir = join(self.output[oracle], "success")
                    pass
                elif self.is_error_msg(results[ERR_CPU_KEY]) != self.is_error_msg(results[ERR_GPU_KEY]):
                    write_dir = join(self.output[oracle], "potential-bug")
                else:
                    write_dir = join(self.output[oracle], "success")
            elif self.is_crash_msg(error):
                write_dir = join(self.output[oracle], "potential-bug")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)
        elif oracle == OracleType.PRECISION:
            code = "import torch\n"
            code += "import time\n"
            code += api.to_code(res=f"results[\"{TIME_LOW_KEY}\"]", low_precision=True)
            code += api.to_diff_code(oracle, res=f"results[\"{TIME_HIGH_KEY}\"]")

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)
            if error == None:
                if isinstance(results[TIME_LOW_KEY], float) and isinstance(results[TIME_HIGH_KEY], float):
                    if results[TIME_LOW_KEY] > self.time_bound * results[TIME_HIGH_KEY] and results[TIME_HIGH_KEY] > self.time_thresold:
                        write_dir = join(self.output[oracle], "potential-bug")
                    else:
                        write_dir = join(self.output[oracle], "success")
                else:
                    write_dir = join(self.output[oracle], "fail")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)

    @staticmethod
    def generate_code(api: TorchAPI, oracle: OracleType) -> str:
        if oracle == OracleType.CRASH:
            return api.to_code()
        elif oracle == OracleType.CUDA:
            code = api.to_code(res="cpu_res", use_try=True)
            code += api.to_diff_code(oracle, res="cuda_res", use_try=True)
            return code
        elif oracle == OracleType.PRECISION:
            code = api.to_code(res="low_res", low_precision=True)
            code += api.to_diff_code(oracle, res="high_res")
            return code
        else:
            assert(0)
    
    @staticmethod
    def run_code(code):
        results = dict()
        results[ERR_CPU_KEY] = None
        results[ERR_GPU_KEY] = None
        error = None
        try:
            exec(code)
        except Exception as e:
            error = str(e)
        return results, error
    
    @staticmethod
    def is_equal(x, y, diff_bound):
        def eq_float_tensor(x, y):
            # not strictly equal
            return torch.allclose(x, y, atol=diff_bound, equal_nan=True)

        x_type = TorchArgument.get_type(x)
        y_type = TorchArgument.get_type(y)
        if x_type != y_type:
            if x_type == ArgType.TORCH_TENSOR and y_type in [ArgType.LIST, ArgType.TUPLE]:
                flag = False
                for temp in y:
                    flag = flag or TorchLibrary.is_equal(x, temp, diff_bound)
                return flag
            elif y_type == ArgType.TORCH_TENSOR and x_type in [ArgType.LIST, ArgType.TUPLE]:
                flag = False
                for temp in x:
                    flag = flag or TorchLibrary.is_equal(y, temp, diff_bound)
                return flag
            return False
        if x_type == ArgType.TORCH_TENSOR:
            x = x.cpu()
            y = y.cpu()
            if x.dtype != y.dtype or x.shape != y.shape:
                return False
            if x.is_sparse:
                x = x.to_dense()
            if y.is_sparse:
                y = y.to_dense()
            if x.is_complex():
                if not y.is_complex(): return False
                return eq_float_tensor(x.real, y.real) and eq_float_tensor(
                    x.imag, y.imag)
            if not x.dtype.is_floating_point:
                return torch.equal(x.cpu(), y.cpu())
            return eq_float_tensor(x, y)
        elif x_type == ArgType.FLOAT:
            return abs(x - y) < diff_bound
        elif x_type in [ArgType.LIST, ArgType.TUPLE]:
            if len(x) != len(y):
                return False
            for i in range(len(x)):
                if TorchLibrary.is_equal(x[i], y[i], diff_bound) == False:
                    return False
            return True
        else:
            return x == y
    
    @staticmethod
    def is_error_msg(error_msg):
        allowed_msgs = ["not implement", "not support"]

        if error_msg == None:
            return False
        for msg in allowed_msgs:
            if msg in error_msg:
                return False
        return True
    
    @staticmethod
    def is_crash_msg(error_msg):
        if error_msg == None:
            return False
        if "INTERNAL ASSERT" in error_msg:
            return True
        else:
            return False
    

def test():
    api_name = "torch.nn.Conv2d"
    api = TorchAPI(api_name)
    MyPytorch = TorchLibrary("torch-torch-output")
    print(MyPytorch.generate_code(api, OracleType.CRASH))
    print(MyPytorch.generate_code(api, OracleType.CUDA))
    print(MyPytorch.generate_code(api, OracleType.PRECISION))
    MyPytorch.test_with_oracle(api, OracleType.CRASH)
    MyPytorch.test_with_oracle(api, OracleType.CUDA)
    MyPytorch.test_with_oracle(api, OracleType.PRECISION)
    # print(TorchArgument.get_type(1))

================================================================================
FILE: src\config\demo_tf.conf
================================================================================
[general]
libs = tf

[mongodb]
# your-mongodb-server
host = 127.0.0.1
# mongodb port
port = 27017 
# name of pytorch database
torch_database = freefuzz-torch
# name of tensorflow database
tf_database = freefuzz-tf

[output]
# output directory for pytorch
torch_output = torch-output
# output directory for tensorflow
tf_output = tf-output

[oracle]
# enable crash oracle
enable_crash = true
# enable cuda oracle
enable_cuda = true
# enable precision oracle
enable_precision = true
# float difference bound: if |a-b| > bound, a is different than b
float_difference_bound = 1e-2
# max time bound: if time(low_precision) > bound * time(high_precision),
# it will be considered as a potential bug
max_time_bound = 10
# only consider the call with time(call) > time_thresold
time_thresold = 1e-3

[mutation]
enable_value_mutation = true
enable_type_mutation = true
enable_db_mutation = true
# the number of times each api is executed
each_api_run_times = 1


================================================================================
FILE: src\config\demo_torch.conf
================================================================================
[general]
libs = torch

[mongodb]
# your-mongodb-server
host = 127.0.0.1
# mongodb port
port = 27017 
# name of pytorch database
torch_database = freefuzz-torch
# name of tensorflow database
tf_database = freefuzz-tf

[output]
# output directory for pytorch
torch_output = torch-output
# output directory for tensorflow
tf_output = tf-output

[torch-output]
# output directory for pytorch
torch-output = torch-output
# output directory for tensorflow
tf_output = tf-output

[oracle]
# enable crash oracle
enable_crash = true
# enable cuda oracle
enable_cuda = true
# enable precision oracle
enable_precision = true
# float difference bound: if |a-b| > bound, a is different than b
float_difference_bound = 1e-5
# max time bound: if time(low_precision) > bound * time(high_precision),
# it will be considered as a potential bug
max_time_bound = 10
# only consider the call with time(call) > time_thresold
time_thresold = 1e-3

[mutation]
enable_value_mutation = true
enable_type_mutation = true
enable_db_mutation = true
# the number of times each api is executed
each_api_run_times = 10

================================================================================
FILE: src\config\expr.conf
================================================================================
[general]
libs = tf,torch

[mongodb]
# your-mongodb-server
host = 127.0.0.1
# mongodb port
port = 27017 
# name of pytorch database
torch_database = freefuzz-torch
# name of tensorflow database
tf_database = freefuzz-tf

[output]
# output directory for pytorch
torch_output = torch-output
# output directory for tensorflow
tf_output = tf-output

[oracle]
# enable crash oracle
enable_crash = true
# enable cuda oracle
enable_cuda = true
# enable precision oracle
enable_precision = true
# float difference bound: if |a-b| > bound, a is different than b
float_difference_bound = 1e-2
# max time bound: if time(low_precision) > bound * time(high_precision),
# it will be considered as a potential bug
max_time_bound = 10
# only consider the call with time(call) > time_thresold
time_thresold = 1e-3

[mutation]
enable_value_mutation = true
enable_type_mutation = true
enable_db_mutation = true
# the number of times each api is executed
each_api_run_times = 1000


================================================================================
FILE: src\constants\enum.py
================================================================================

from enum import IntEnum
class OracleType(IntEnum):
    CRASH = 1
    CUDA = 2
    PRECISION = 3

================================================================================
FILE: src\constants\keys.py
================================================================================
# execution result
RES_KEY = "results"
RESULT_KEY = "res"
ERROR_KEY = "err"
ERR_ARG_KEY = "error_args"
ERR_CPU_KEY = "err_cpu"
ERR_GPU_KEY = "err_gpu"
RES_CPU_KEY = "res_cpu"
RES_GPU_KEY = "res_gpu"
ERR_HIGH_KEY = "err_high"
ERR_LOW_KEY = "err_low"
RES_HIGH_KEY = "res_high"
RES_LOW_KEY = "res_low"
TIME_LOW_KEY = "time_low"
TIME_HIGH_KEY = "time_high"


================================================================================
FILE: src\instrumentation\tensorflow\decorators.py
================================================================================
from tensorflow.instrumentation.signature_handler import SignatureHandler
from tensorflow.instrumentation.write_tools import write_fn
import os
import json
sighdl = SignatureHandler()

def isiterable(t):
    return isinstance(t, list) or isinstance(t, tuple)

def get_signature_for_tensors(t):
    return sighdl.get_var_signature(t)


def build_param_dict(*args, **kwargs):
    param_dict = dict()
    for ind, arg in enumerate(args):
        param_dict['parameter:%d' % ind] = sighdl.get_var_signature(arg)
    for key, value in kwargs.items():
        if key == 'name': continue
        param_dict[key] = sighdl.get_var_signature(value)
    param_dict = dict(param_dict)
    return param_dict




def dump_signature_of_class(klass, class_name, output_dir):
    if not hasattr(klass, '__call__'):
        return klass
    old_init = klass.__init__
    old_call = klass.__call__
    init_params = dict()


    def new_init(self, *args, **kwargs):
        nonlocal init_params
        try:
            init_params = build_param_dict(*args, **kwargs)
        except Exception as e:
            print(e.message)
        old_init(self, *args, **kwargs)

    def new_call(self, *inputs, **kwargs):
        nonlocal init_params

        input_signature = get_signature_for_tensors(inputs)
        outputs = old_call(self, *inputs, **kwargs)
        output_signature = get_signature_for_tensors(outputs)
        write_fn(self.__class__.__module__ + '.' + self.__class__.__name__, init_params, input_signature,
                 output_signature)
        return outputs

    klass.__init__ = new_init
    klass.__call__ = new_call
    return klass

from functools import wraps


def dump_signature_of_function(func, hint, output_dir):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import json
        import os


        outputs = func(*args, **kwargs)
        output_signature = get_signature_for_tensors(outputs)
        param_dict = build_param_dict(*args, **kwargs)
        write_fn(hint, param_dict, None, output_signature)
        return outputs

    if not callable(func):
        return func

    return wrapper


================================================================================
FILE: src\instrumentation\tensorflow\hijack.py
================================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import inspect
from tensorflow.instrumentation.decorators import dump_signature_of_class, dump_signature_of_function
def hijack(output_dir="signature_db"):
    hijack_all(output_dir)


def hijack_api(obj, func_name_str, output_dir):
    """
    Function to hijack an API.

    Args:
        obj: the base module. This function is currently specific to TensorFlow.
            So obj should be tensorflow.
        func_name_str: A string. The full name of the api (except 'tf.'). For example, the name of
            `tf.keras.losses.MeanSquaredError` should be 'keras.losses.MeanSquaredError'.

    Returns:
        A boolean, indicating if hijacking is successful.


    The targeted API can be either a function or a class (the type will be detected by this function).
    This function would replace the original api with the new decorated api we created. This is achieved
    in a fairly simple and straight-forward way. For the example above, we just set the attribute by calling
    `setattr(tf.keras.losses, 'MeanSquaredError', wrapped_func)`.
    """
    func_name_list = func_name_str.split('.')
    func_name = func_name_list[-1]

    # Get the module object and the api object.
    module_obj = obj
    if len(func_name_list) > 1:
        for module_name in func_name_list[:-1]:
            module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    # Utilities.
    def is_class(x):
        return inspect.isclass(x)
    def is_callable(x):
        return callable(x)
    def is_built_in_or_extension_type(x):
        if is_class(x) and hasattr(x, '__dict__') and not '__module__' in x.__dict__:
            return True
        else:
            return False
    # Handle special cases of types.
    if is_built_in_or_extension_type(orig_func):
      return False
    if is_class(orig_func):
        if hasattr(orig_func, '__slots__'):
            return False
        wrapped_func = dump_signature_of_class(orig_func, func_name_str, output_dir=output_dir)
        setattr(module_obj, func_name, wrapped_func)
        return True
    else:
      if is_callable(orig_func):
        wrapped_func = dump_signature_of_function(orig_func, func_name_str, output_dir=output_dir)
        setattr(module_obj, func_name, wrapped_func)
        return True
      else:
        return False

def should_skip(api):

    skip_list = [
        'tf.keras.layers.Layer',
        'tf.compat.v1.keras.layers.Layer',
        'tf.Module',
        'tf.compat.v1.Module',
        'tf.compat.v1.flags.FLAGS',
        'tf.compat.v1.app.flags.EnumClassListSerializer',
        'tf.compat.v1.app.flags.EnumClassSerializer',
        'tf.compat.v1.flags.EnumClassListSerializer',
        'tf.compat.v1.flags.EnumClassSerializer',
        'tf.init_scope',
        'tf.TensorShape',
        'tf.Variable',
        'tf.compat.v1.Variable',
        'tf.ResourceVariable',
        'tf.Tensor',
        'tf.compat.v1.Tensor',
        'tf.compat.v1.flags.tf_decorator.make_decorator',
        'tf.compat.v1.flags.tf_decorator.tf_stack.extract_stack',
        'tf.compat.v1.flags.tf_decorator.unwrap',
        'tf.compat.v1.flags.tf_decorator.rewrap',
        'tf.compat.v1.app.flags.tf_decorator.make_decorator',
        'tf.compat.v1.app.flags.tf_decorator.rewrap',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.CurrentModuleFilter',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.FrameSummary',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.StackSummary',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.StackTraceFilter',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.StackTraceMapper',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.StackTraceTransform',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.extract_stack',
        'tf.compat.v1.app.flags.tf_decorator.unwrap',

    ]
    skip_key_word = [
        'tf.compat.v1',
        'tf.debugging',
        'tf.distribute',
        'tf.errors',
        'tf.profiler',
        'tf.test',
        'tf.tpu',
        'tf.summary',
        'tpu',
        'TPU',
        # 'tf.quantization', 
        # 'tf.experimental.numpy',

    ]
    
    if api.find('tf.') != 0:
        return True
    # Skip the current api if it's in the skip list.
    if api in skip_list:
        return True
    # Skip the current api if it has some keywords.
    for kw in skip_key_word:
        if kw in api:
            return True

def hijack_all(output_dir, verbose=False):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success_list = []
    failed_list = []
    skip_list = []
    import os
    api_file = __file__.replace("hijack.py", "api_list.txt")
    with open(api_file, 'r') as fr:
        apis = fr.readlines()
    print('Number of total apis: ', len(apis)) 
    skip_apis = False
    cnt = 0
    for i, api in enumerate(apis):
        api = api.strip()
        if skip_apis:
            if should_skip(api):
                skip_list.append(api + "\n")
                continue

        hijack_api(tf, api[3:], output_dir)


================================================================================
FILE: src\instrumentation\tensorflow\signature_handler.py
================================================================================
import tensorflow as tf
import numpy as np

import json
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

def is_iterable(v):
    return isinstance(v, list) or isinstance(v, tuple)
def get_var_class_full_name(v):
    return v.__class__.__module__ + '.' + v.__class__.__name__

def json_serialize_value(v):
    """Return the json serializable value of v. """
    try:
        return json.dumps(v)
    except Exception as e:
        return str(v)

def json_deserialize_value(v):
    """Return the json serializable value of v. """
    try:
        return json.loads(v)
    except Exception as e:
        return v

class SignatureHandler:
    python_built_in_types = [str,
                             int, float, complex,
                             list, tuple, range,
                             dict,
                             set, frozenset,
                             bool,
                             bytes, bytearray, memoryview]
    def get_var_signature(self, v):
        if self.check_var_tensor(v):
            return self.get_tensor_signature(v)
        if self.check_var_variable(v):
            return self.get_variable_signature(v)
        if self.check_var_nparray(v):
            return self.get_nparray_signature(v)
        if self.check_var_tf_object(v):
            return self.get_tf_object_signature(v)

        if self.check_var_list(v):
            return self.get_list_signature(v)
        
        if self.check_var_raw(v):
            return self.get_raw_signature(v)
            
        return self.get_other_signature(v)

    def check_var_raw(self, v):
        """ Check if a variable is a python built-in object. """
        if type(v) in self.python_built_in_types:
            return True
        else:
            return False

    def get_raw_signature(self, v):
        s = dict()
        s['Label'] = 'raw'
        s['value'] = json_serialize_value(v)
        return s


    def check_var_list(self, v):
        """ Check if a variable is a list. """
        return isinstance(v, list) or isinstance(v, tuple)

    def get_list_signature(self, v):
        s = dict()
        s['Label'] = 'list'
        s['value'] = [self.get_var_signature(e) for e in v]
        return s

    def check_var_tuple(self, v):
        """ Check if a variable is a list. """
        return isinstance(v, list)

    def get_tuple_signature(self, v):
        s = dict()
        s['Label'] = 'tuple'
        s['value'] = (self.get_var_signature(e) for e in v)
        return s

    def check_var_tensor(self, v):
        """ Check if a variable is a TensorFlow tensor """
        if isinstance(v, tf.Tensor) or isinstance(v, KerasTensor):
            return True
        else:
            return False
    
    def check_var_variable(self, v):
        """ Check if a variable is a TensorFlow variable """
        if isinstance(v, tf.Variable):
            return True
        else:
            return False

    def get_tensor_shape(self, v):
        assert isinstance(v.shape, tf.TensorShape)
        try:
            return v.shape.as_list()
        except ValueError:
            # s has unknown shape (unknown rank): TensorShape(None)
            return None

    def get_tensor_signature(self, v):
        """ v is a Tensor."""
        s = dict()
        s['Label'] = 'tensor'
        if isinstance(v, KerasTensor):
            s['Label'] = 'KerasTensor'
        assert isinstance(v.dtype, tf.dtypes.DType)
        s['dtype'] = v.dtype.name
        s['shape'] = self.get_tensor_shape(v)
        return s

    def get_variable_signature(self, v):
        """ v is a variable"""
        s = dict()
        s['Label'] = 'variable'
        assert isinstance(v.dtype, tf.dtypes.DType)
        s['dtype'] = v.dtype.name
        s['shape'] = self.get_tensor_shape(v)
        return s

    def check_var_nparray(self, v):
        return isinstance(v, np.ndarray)

    def get_nparray_signature(self, v):
        s = dict()
        s['Label'] = 'nparray'
        s['shape'] = v.shape
        s['dtype'] = v.dtype.name
        return s

    def check_var_tf_object(self, v):
        return 'tensorflow' in get_var_class_full_name(v)

    def get_tf_object_signature(self, v):
        s = dict()
        s['Label'] = 'tf_object'
        s['class_name'] = get_var_class_full_name(v)
        if s["class_name"] == "tensorflow.python.framework.dtypes.DType":
            s['to_str'] = str(v)
        if hasattr(v, 'shape'):
            s['shape'] = self.get_tensor_shape(v)
        if hasattr(v, 'dtype'):
            if isinstance(v.dtype, tf.dtypes.DType):
                s['dtype'] = v.dtype.name
            else:
                s['type'] = str(v.dtype)

        return s

    def get_other_signature(self, v):
        s = dict()
        s['Label'] = 'other'
        s['type'] = str(type(v))
        return s



================================================================================
FILE: src\instrumentation\tensorflow\write_tools.py
================================================================================
import pymongo

"""
You should configure the database
"""
tf_db = pymongo.MongoClient(host="localhost", port=27017)["freefuzz-tf"]

def write_fn(func_name, params, input_signature, output_signature):
    params = dict(params)
    out_fname = "tf." + func_name
    if input_signature != None:
        params['input_signature'] = input_signature
    params['output_signature'] = output_signature
    tf_db[out_fname].insert_one(params)


================================================================================
FILE: src\instrumentation\tensorflow\__init__.py
================================================================================


================================================================================
FILE: src\instrumentation\torch\decorate_cls.py
================================================================================
import json
from write_tools import write_fn

def decorate_class(klass, hint):
    if not hasattr(klass, '__call__'):
        return klass
    old_init = klass.__init__
    old_call = klass.__call__
    init_params = dict()

    def json_serialize(v):
        try:
            json.dumps(v)
            return v
        except Exception as e:
            if hasattr(v, '__name__'):
                return v.__name__
            elif hasattr(v, '__class__'):
                res = []
                if isinstance(v, tuple) or isinstance(v, list):
                    for vi in v:
                        if hasattr(vi, 'shape'):
                            res.append(get_var_signature(vi))
                        elif isinstance(vi, tuple) or isinstance(vi, list):
                            res2 = []
                            for vii in vi:
                                if (hasattr(vii, 'shape')):
                                    res2.append(get_var_signature(vii))
                            res.append(res2)
                    return res
                else:
                    return v.__class__.__module__ + v.__class__.__name__
            return str(type(v))

    def build_param_dict(*args, **kwargs):
        param_dict = dict()
        for ind, arg in enumerate(args):
            param_dict['parameter:%d' % ind] = json_serialize(arg)
        for key, value in kwargs.items():
            param_dict[key] = json_serialize(value)
        return dict(param_dict)

    def get_var_shape(var):
        if hasattr(var, 'shape'):
            s = var.shape
            if isinstance(s, list):
                return s
            elif isinstance(s, tuple):
                return list(s)
            else:
                try:
                    return list(s)  # convert torch.Size to list
                except Exception as e:
                    print(e.message)

    def get_var_dtype(var):
        if hasattr(var, 'dtype'):
            return str(var.dtype)  # string
        if isinstance(var, list):
            res = '['
            for varx in var:
                res += type(varx).__name__ + ","
            return res[:-1] + "]"  # remove the ending ","
        elif isinstance(var, tuple):
            res = '['
            for varx in var:
                res += type(varx).__name__ + ","
            return res[:-1] + "]"
        else:
            try:
                return type(var).__name__
            except Exception as e:
                print(e.message)

    def get_shape_for_tensors(t):
        if isinstance(t, list) or isinstance(t, tuple):
            input_shape = [get_var_shape(i) for i in t]
        else:
            input_shape = get_var_shape(t)
        return input_shape

    def get_var_signature(var):
        s = dict()
        s['shape'] = get_var_shape(var)
        s['dtype'] = get_var_dtype(var)
        return s

    def get_signature_for_tensors(t):
        if isinstance(t, list) or isinstance(t, tuple):
            signatures = [get_var_signature(i) for i in t]
        else:
            signatures = get_var_signature(t)
        return signatures

    def new_init(self, *args, **kwargs):
        nonlocal init_params
        init_params = build_param_dict(*args, **kwargs)
        old_init(self, *args, **kwargs)

    def new_call(self, *inputs, **kwargs):
        nonlocal init_params
        input_signature = get_signature_for_tensors(inputs)
        outputs = old_call(self, *inputs, **kwargs)
        output_signature = get_signature_for_tensors(outputs)
        write_fn(hint, dict(init_params), input_signature, output_signature)
        return outputs

    klass.__init__ = new_init
    klass.__call__ = new_call
    return klass


================================================================================
FILE: src\instrumentation\torch\decorate_func.py
================================================================================
from functools import wraps
import json
import os
from write_tools import write_fn


def decorate_function(func, hint):
    @wraps(func)
    def wrapper(*args, **kwargs):
        def json_serialize(v):
            """Return a json serializable object. """
            try:
                json.dumps(v)
                return v  # v is a int, float, list, ...
            except Exception as e:
                if hasattr(v, 'shape'):  # v numpy array
                    return get_var_signature(
                        v)  # A dict of signature {'shape':..., 'type':...}
                if hasattr(v, '__name__'):  #  v is a function
                    return v.__name__
                elif hasattr(v, '__class__'):  # v is a class
                    res = []
                    if isinstance(v, tuple) or isinstance(v, list):
                        for vi in v:
                            if hasattr(vi, 'shape'):
                                res.append(get_var_signature(vi))
                            elif isinstance(vi, tuple) or isinstance(vi, list):
                                res2 = []
                                for vii in vi:
                                    if (hasattr(vii, 'shape')):
                                        res2.append(get_var_signature(vii))
                                res.append(res2)
                        return res
                    else:
                        return v.__class__.__module__ + v.__class__.__name__  # v.name
                else:
                    raise Exception('Error [json serialize ] %s' % v)

        def build_param_dict(*args, **kwargs):
            param_dict = dict()
            for ind, arg in enumerate(args):
                param_dict['parameter:%d' % ind] = json_serialize(arg)
            for key, value in kwargs.items():
                param_dict[key] = json_serialize(value)
            return param_dict

        def get_var_shape(var):
            if hasattr(var, 'shape'):  # var is numpy.ndarray or tensor
                s = var.shape
                if isinstance(s, list):
                    return s
                elif isinstance(s, tuple):
                    return list(s)
                else:
                    try:
                        return list(s)  # convert torch.Size to list
                    except Exception as e:
                        print(e.message)

        def get_var_dtype(var):
            if hasattr(var, 'dtype'):
                return str(var.dtype)  # string
            if isinstance(var, list):
                res = '['
                for varx in var:
                    res += type(varx).__name__ + ","
                return res[:-1] + "]"  # remove the ending ","
            elif isinstance(var, tuple):
                res = '['
                for varx in var:
                    res += type(varx).__name__ + ","
                return res[:-1] + "]"
            else:
                try:
                    return type(var).__name__
                except Exception as e:
                    print(e.message)

        def get_shape_for_tensors(t):
            if isinstance(t, list):
                input_shape = [get_var_shape(i) for i in t]
            else:
                input_shape = get_var_shape(t)
            return input_shape

        def get_var_signature(var):
            s = dict()
            s['shape'] = get_var_shape(var)
            s['dtype'] = get_var_dtype(var)
            return s

        def get_signature_for_tensors(t):
            if isinstance(t, list):
                signatures = [get_var_signature(i) for i in t]
            else:
                signatures = get_var_signature(t)
            return signatures

        outputs = func(*args, **kwargs)
        output_signature = get_signature_for_tensors(outputs)
        param_dict = build_param_dict(*args, **kwargs)
        write_fn(hint, param_dict, None, output_signature)
        return outputs

    if not callable(func):
        return func

    return wrapper


================================================================================
FILE: src\instrumentation\torch\write_tools.py
================================================================================
import pymongo

"""
You should configure the database
"""
torch_db = pymongo.MongoClient(host="localhost", port=27017)["freefuzz-torch"]

def write_fn(func_name, params, input_signature, output_signature):
    params = dict(params)
    out_fname = "torch." + func_name
    params['input_signature'] = input_signature
    params['output_signature'] = output_signature
    torch_db[out_fname].insert_one(params)

================================================================================
FILE: src\instrumentation\torch\__init__.py
================================================================================
import torch.nn.utils.prune

import decorate_function
import decorate_class
import inspect

def hijack(obj, func_name_str, mode=""):
    func_name_list = func_name_str.split('.')
    func_name = func_name_list[-1]

    module_obj = obj
    if len(func_name_list) > 1:
        for module_name in func_name_list[:-1]:
            module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    def is_class(x):
        return inspect.isclass(x)
    def is_callable(x):
        return callable(x)

    if mode == "function":
        wrapped_func = decorate_function(orig_func, func_name_str)
    elif mode == "class":
        wrapped_func = decorate_class(orig_func, func_name_str)
    else:
        if is_class(orig_func):
            wrapped_func = decorate_class(orig_func, func_name_str)
        elif is_callable(orig_func):
            wrapped_func = decorate_function(orig_func, func_name_str)
        else:
            wrapped_func = orig_func
    setattr(module_obj, func_name, wrapped_func)


with open(__file__.replace("__init__.py", "torch.txt"), "r") as f1:
    lines = f1.readlines()
    skipped = ["enable_grad", "get_default_dtype", "load", "tensor", "no_grad", "jit"]
    for l in lines:
        l = l.strip()
        if l not in skipped:
            hijack(torch, l, mode="function")

with open(__file__.replace("__init__.py", "torch.nn.txt"), "r") as f2:
    lines = f2.readlines()
    for l in lines:
        l = l.strip()
        hijack(torch, l)

with open(__file__.replace("__init__.py", "torch.nn.functional.txt"), "r") as f3:
    lines = f3.readlines()
    for l in lines:
        l = l.strip()
        hijack(torch, l, "function")


================================================================================
FILE: src\preprocess\process_data.py
================================================================================
import pymongo
import textdistance
import re
import numpy as np
import configparser
import sys
from os.path import join

signature_collection = "signature"
similarity_collection = "similarity"

"""
Similarity Part:
This part relys on the database so I put it into this file
"""

API_def = {}
API_args = {}


def string_similar(s1, s2):
    return textdistance.levenshtein.normalized_similarity(s1, s2)


def loadAPIs(api_file='../data/torch_APIdef.txt'):
    global API_def, API_args
    with open(api_file, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            API_name = line.split("(")[0]
            API_args_match = re.search("\((.*)\)", line)
            try:
                API_args_text = API_args_match.group(1)
            except:
                # with open("log/tf/api_def_error.txt", 'a') as f:
                #     f.write(line + "\n")
                # continue
                raise ValueError(line)
            # print(API_args_text)
            if API_name not in API_def.keys():
                API_def[API_name] = line
                API_args[API_name] = API_args_text


def query_argname(arg_name):
    '''
    Return a list of APIs with the exact argname
    '''
    def index_name(api_name, arg_name):
        arg_names = DB[signature_collection].find_one({"api": api_name})["args"]
        for idx, name in enumerate(arg_names):
            if name == arg_name:
                return f"parameter:{idx}"
        return None

    APIs = []
    for api_name in API_args.keys():
        # search from the database
        # if arg_name exists in the records of api_name, append api_name into APIs
        if api_name not in DB.list_collection_names(
        ) or arg_name not in API_args[api_name]:
            continue
        temp = DB[api_name].find_one({arg_name: {"$exists": True}})
        if temp == None:
            # since there are two forms of names for one argument, {arg_name} and parameter:{idx}
            # we need to check the parameter:{idx}
            idx_name = index_name(api_name, arg_name)
            if idx_name and DB[api_name].find_one({idx_name: {"$exists": True}}):
                APIs.append(api_name)
        else:
            APIs.append(api_name)
    return APIs


def mean_norm(x):
    return (x - np.mean(x)) / (np.max(x) - np.min(x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def similarAPI(API, argname):
    '''
    Return a list of similar APIs (with the same argname) and their similarities
    '''
    API_with_same_argname = query_argname(argname)
    if len(API_with_same_argname) == 0:
        return [], []
    probs = []
    original_def = API_def[API]
    for item in API_with_same_argname:
        to_compare = API_def[item]
        probs.append(string_similar(original_def, to_compare))
    prob_norm2 = softmax(probs)
    return API_with_same_argname, prob_norm2




"""
Writing Parts (Data Preprocessing):
This part is to write API signature, argument value space and similarity
to database.

You SHOULD call these functions in this order!
    1. write_API_signature
    2. write_similarity
"""

def write_API_signature(library_name='torch'):
    """
    API's signature will be stored in 'signature' collection with the form
        api: the name of api
        args: the list of arguments' names
    """
    names = DB.list_collection_names()
    for api_name in names:
        if not api_name.startswith(library_name):
            continue
        if api_name not in API_args.keys():
            DB[signature_collection].insert_one({"api": api_name, "args": []})
            continue

        arg_names = []
        for temp_name in API_args[api_name].split(","):
            temp_name = temp_name.strip()
            if len(temp_name) == 0 or temp_name == "*":
                continue
            if "=" in temp_name:
                temp_name = temp_name[:temp_name.find("=")]
            arg_names.append(temp_name)
        DB[signature_collection].insert_one({
            "api": api_name,
            "args": arg_names
        })


def write_similarity(library_name='torch'):
    """
    Write the similarity of (api, arg) in 'similarity' with the form:
        api: the name of api
        arg: the name of arg
        APIs: the list of similar APIs
        probs: the probability list
    """
    names = DB.list_collection_names()
    for api_name in names:
        if not api_name.startswith(library_name):
            continue

        print(api_name)
        arg_names = DB["signature"].find_one({"api": api_name})["args"]
        for arg_name in arg_names:
            APIs, probs = similarAPI(api_name, arg_name)
            sim_dict = {}
            sim_dict["api"] = api_name
            sim_dict["arg"] = arg_name
            sim_dict["APIs"] = APIs
            sim_dict["probs"] = list(probs)
            DB[similarity_collection].insert_one(sim_dict)


if __name__ == "__main__":
    target = sys.argv[1]
    if target not in ["torch", "tf"]:
        print("Only support 'torch' or 'tf'!")
        assert(0)

    """
    Database Settings
    """
    config_name = f"demo_{target}.conf"
    freefuzz_cfg = configparser.ConfigParser()
    freefuzz_cfg.read(join("config", config_name))

    # database configuration
    mongo_cfg = freefuzz_cfg["mongodb"]
    host = mongo_cfg["host"]
    port = int(mongo_cfg["port"])

    DB = pymongo.MongoClient(host, port)[mongo_cfg[f"{target}_database"]]

    loadAPIs(join("..", "data", f"{target}_APIdef.txt"))
    write_API_signature(target)
    write_similarity(target)


================================================================================
FILE: src\utils\converter.py
================================================================================
def str_to_bool(s):
    return True if s.lower() == 'true' else False


================================================================================
FILE: src\utils\printer.py
================================================================================
def dump_data(content, file_name, mode="w"):
    with open(file_name, mode) as f:
        f.write(content)

================================================================================
FILE: src\utils\probability.py
================================================================================
from numpy.random import rand

def choose_from_list() -> bool:
    return rand() < 0.2

def change_tensor_dimension() -> bool:
    return rand() < 0.3

def add_tensor_dimension() -> bool:
    return rand() < 0.5

def change_tensor_shape() -> bool:
    return rand() < 0.3

def change_tensor_dtype() -> bool:
    return rand() < 0.3

def do_type_mutation() -> bool:
    return rand() < 0.2

def do_select_from_db() -> bool:
    return rand() < 0.2


================================================================================
FILE: src\utils\skip.py
================================================================================

with open(__file__.replace("utils/skip.py", "config/skip_torch.txt")) as f:
    skip_torch = f.read().split("\n")

with open(__file__.replace("utils/skip.py", "config/skip_tf.txt")) as f:
    skip_tf = f.read().split("\n")

def need_skip_torch(api_name):
    if api_name in skip_torch:
        return True
    else:
        return False

def need_skip_tf(api_name):
    if api_name in skip_tf:
        return True
    skip_keywords = ["tf.keras.applications", "Input", "get_file"]
    for keyword in skip_keywords:
        if keyword in api_name:
            return True
    return False
