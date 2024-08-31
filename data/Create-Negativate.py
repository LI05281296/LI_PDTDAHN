import math
import random

import pandas as pd
import numpy as np


def partition(ls, size):
    return [ls[i:i + size] for i in range(0, len(ls), size)]

data = pd.read_csv('..\\mashup_data\\pos.csv', header=None)

def NegativeGenerate(DrugDiseaseTarget, DrugDisease, DrugTarget, DiseaseTarget, AllDurg, AllDisease, AllTarget):
    import random
    NegativeSample = []
    counterN = 0
    # 进入一个循环，直到生成的负样本数量达到与输入的DrugDisease列表相同的数量
    while counterN < len(DrugDiseaseTarget):
        #  随机选择一个药物和一个疾病和一个target，构成一个药物-疾病-target对
        counterDR = random.randint(0, len(AllDurg) - 1)
        counterDI = random.randint(0, len(AllDisease) - 1)
        counterTA = random.randint(0, len(AllTarget) - 1)
        # drug、disease、target 的相互作用
        DrugAndDiseaseAndTargetPair = []
        DrugAndDiseaseAndTargetPair.append(AllDurg[counterDR])
        DrugAndDiseaseAndTargetPair.append(AllDisease[counterDI])
        DrugAndDiseaseAndTargetPair.append(AllTarget[counterTA])
        # drug、disease没有相互作用
        DrugAndDiseasePair = []
        DrugAndDiseasePair.append(AllDurg[counterDR])
        DrugAndDiseasePair.append(AllDisease[counterDI])
        # drug、target没有相互作用
        DrugAndTargetPair = []
        DrugAndTargetPair.append(AllDurg[counterDR])
        DrugAndTargetPair.append(AllTarget[counterTA])
        # disease、target没有相互作用
        DiseaseAndTargetPair = []
        DiseaseAndTargetPair.append(AllDisease[counterDI])
        DiseaseAndTargetPair.append(AllTarget[counterTA])
        #  flag1检查这个药物-疾病-靶对是否已经存在于输入的DrugDiseaseTarget列表中
        #  flag2检查这个药物-疾病对是否已经存在于输入的DrugDisease列表中
        #  flag3检查这个药物-靶对是否已经存在于输入的DrugTarget列表中
        #  flag4检查这个疾病-靶对是否已经存在于输入的DiseaseTarget列表中
        flag1 = 0
        flag2 = 0
        flag3 = 0
        flag4 = 0
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0

        while counter1 < len(DrugDiseaseTarget):
            if DrugAndDiseaseAndTargetPair == DrugDiseaseTarget[counter1]:
                flag1 = 1
                break
            counter1 = counter1 + 1
            #  flag1为1（已存在于DrugDiseaseTarget列表），则跳过当前循环，不添加到NegativeSample中
            if flag1 == 1:
                continue
        while counter2 < len(DrugDisease):
            if DrugAndDiseasePair == DrugDisease[counter2]:
                flag2 = 1
                break
            counter2 = counter2 + 1
            #  flag2为1（已存在于DrugDisease列表），则跳过当前循环，不添加到NegativeSample中
            if flag2 == 1:
                continue
        while counter3 < len(DrugTarget):
            if DrugAndTargetPair == DrugTarget[counter3]:
                flag3 = 1
                break
            counter3 = counter3 + 1
            #  flag3为1（已存在于DrugDiseaseTarget列表），则跳过当前循环，不添加到NegativeSample中
            if flag3 == 1:
                continue
        while counter4 < len(DiseaseTarget):
            if DiseaseAndTargetPair == DiseaseTarget[counter4]:
                flag4 = 1
                break
            counter4 = counter4 + 1
            #  flag4为1（已存在于DrugDiseaseTarget列表），则跳过当前循环，不添加到NegativeSample中
            if flag4 == 1:
                continue

        #  flag5用于检查这个药物-疾病对是否已经存在于NegativeSample中
        flag5 = 0
        counter5 = 0
        while counter5 < len(NegativeSample):
            if DrugAndDiseaseAndTargetPair == NegativeSample[counter5]:
                flag5 = 1
                break
            counter5 = counter5 + 1
            if flag5 == 1:
                continue
        # 三者都没有关系，加入到负样本
        if (flag1 == 0 and flag2 == 0 and flag3 == 1 and flag4 == 1 and flag5 == 0):
            NamePair = []
            NamePair.append(AllDurg[counterDR])
            NamePair.append(AllDisease[counterDI])
            NamePair.append(AllTarget[counterTA])
            NegativeSample.append(NamePair)
            counterN = counterN + 1
    return NegativeSample

node_list = pd.read_csv('node_num.csv', header=None, names=['node', 'num'])
drug_list = node_list[:124]
disease_list = node_list[124:301]
target_list = node_list[301:]
triple_data = pd.read_csv('pos.csv', header=None, names=['drug', 'disease', 'target', 'label'])
triple_data = triple_data[['drug', 'disease', 'target']]
drug_disease = pd.read_csv('drug_disease_num.csv', header=None, names=['drug', 'disease'])
drug_target = pd.read_csv('drug_target_num.csv', header=None, names=['drug', 'target'])
disease_target = pd.read_csv('disease_target_num.csv', header=None, names=['disease', 'target'])
NegativeSample = NegativeGenerate(triple_data.values.tolist(),
                                  drug_disease.values.tolist(),
                                  drug_target.values.tolist(),
                                  disease_target.values.tolist(),
                                  drug_list['num'].values.tolist(),
                                  disease_list['num'].values.tolist(),
                                  target_list['num'].values.tolist())
NegativeSample = pd.DataFrame(NegativeSample)

NegativeSample['label'] = 0
NegativeSample.to_csv('neg6_.csv', header=None, index=False)