import argparse

parer = argparse.ArgumentParser(description="COISAS")

parer.add_argument('--type', type=str,required=True,help='modelo')

args = parer.parse_args()

from Utils.Results import gets_models_result, print_results


res = gets_models_result("Model4/XGboost","XG Boost")

print_results(res,args.type)
