import os

def listar_arquivos(pasta):
    arquivos = []
    # Percorre todos os arquivos na pasta
    for nome_arquivo in os.listdir(pasta):
        # Verifica se o caminho é um arquivo
        if os.path.isfile(os.path.join(pasta, nome_arquivo)):
            arquivos.append(nome_arquivo)
    return arquivos
def get_value(type, file):
    dict = {"0": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
            "1": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}, "accuracy": 0.0, "name":""}

    dict["name"] = get_name(type, file)
    # Abre o arquivo em modo de leitura
    with (open("Model/"+type+"/"+file, 'r') as arquivo):
        # Percorre linha por linha
        get = False
        ofensive = True
        nonOfensive = True
        for linha in arquivo:

            if linha.rstrip().split() == []:
                continue

            line = linha.rstrip().split()

            if get:
                if nonOfensive and float(line[0]) == 0.0:
                    dict["0"]["precision"] = float(line[1])
                    dict["0"]["recall"] = float(line[2])
                    dict["0"]["f1_score"] = float(line[3])
                    nonOfensive = False

                if ofensive and float(line[0]) == 1.0:
                    dict["1"]["precision"] = float(line[1])
                    dict["1"]["recall"] = float(line[2])
                    dict["1"]["f1_score"] = float(line[3])
                    ofensive = False
                    
                if line[0] == 'accuracy':
                    dict["accuracy"] = float(line[1])
            if "clasificador report:" in linha.rstrip():
                get = True
        return dict

def get_name(model,file):
    name = file.replace("best_model_", "")
    name = name.replace(".txt", "")
    name = name.replace("_"," ")
    name = model + " " + name

    return name

def gets_models_result(path,type):
    res = listar_arquivos(path)
    result = []

    for file in res:
        result.append(get_value(type,file))

    return result
def print_results(results):
    print('\n\n')
    for res in results:
        print("Model: "+res["name"])
        print("Accuracy: "+str(res["accuracy"]))
        print("\t    Precission\t       Recall\tF1-Score")
        print("0:\t\t"+"{:.2f}".format(res['0']['precision'])+"\t\t"+"{:.2f}".format(res['0']['recall'])+"\t  "+"{:.2f}".format(res['0']['f1_score']))
        print("1:\t\t"+"{:.2f}".format(res['1']['precision'])+"\t\t"+"{:.2f}".format(res['1']['recall'])+"\t  "+"{:.2f}".format(res['1']['f1_score']))
        print("\n")

