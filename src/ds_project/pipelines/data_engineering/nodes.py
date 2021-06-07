# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example nodes to solve some common data engineering problems using PySpark,
such as:
* Extracting, transforming and selecting features
* Split data into training and testing datasets
"""

from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder

raw_feature_columns = [
        'OpSys', 
        'EdLevel', 
        'MainBranch' , 
        'Country', 
        'JobSeek', 
        'YearsCode']
raw_target_column = "compAboveAvg"
strIndexOutputCols = ['stringindexed_' + c for c in raw_feature_columns]
oneHotOutputCols = ['onehot_' + c for c in raw_feature_columns]

def extract_columns(data: DataFrame) -> DataFrame:
    data = data.withColumn("compAboveAvg", (col("convertedComp") > 65000).cast("String"))
    data = data.filter("compAboveAvg is not NULL")

    transformed_data = data.select(raw_feature_columns + [raw_target_column])
    return transformed_data

def apply_string_indexer(data: DataFrame) -> DataFrame:
    strIndexInputCols = raw_feature_columns

    indexer = StringIndexer(inputCols=strIndexInputCols, outputCols=strIndexOutputCols).setHandleInvalid("keep")
    transformed_data = (
        indexer.fit(data)
        .transform(data)
        #.drop(raw_target_column)
    )
    return transformed_data

def apply_onehot_encoding(data: DataFrame) -> DataFrame:   
    onehot = OneHotEncoder(inputCols=strIndexOutputCols, outputCols=oneHotOutputCols)
    transformed_data = (
        onehot.fit(data)
        .transform(data)
        #.drop(raw_target_column)
    )
    return transformed_data

def apply_vector_assembler(data: DataFrame) -> DataFrame:   
    # merge  feature columns into a single features vector column
    vector_assembler = VectorAssembler(
        inputCols=oneHotOutputCols, outputCol="features"
    )
    transformed_data = vector_assembler.transform(data)
    return transformed_data

def apply_string_indexer_on_label (data: DataFrame) -> DataFrame:   
    # convert the textual representation of the species into numerical label column
    indexer = StringIndexer(inputCol=raw_target_column, outputCol="label")
    transformed_data = (
        indexer.fit(data)
        .transform(data)
        #.drop(raw_target_column)
    )
    return transformed_data

def split_data(
    transformed_data: DataFrame, example_test_data_ratio: float
) -> List[DataFrame]:
    """Node for splitting the data set into training and test
    sets, each split into features and labels.
    The split ratio parameter is taken from conf/base/parameters.yml.
    """
    example_train_data_ratio = 1 - example_test_data_ratio
    training_data, testing_data = transformed_data.randomSplit(
        [example_train_data_ratio, example_test_data_ratio]
    )
    return [training_data, testing_data]
