from BoOp import BoParas
import yaml
import sys
from logging import Logger


"""
    All result can be seen in ./Loggging
    ./Loggging/main shows all info. of runs;
    ./Loggging/ML show ML runs, specifically, *main.json shows all BoOpt re.s on given range of paras but *PA.json shows re.s of given paras. 

"""


if __name__ == '__main__':

    sys.stdout = Logger()
    f = open(r"template.yaml")
    config = yaml.load(f, Loader=yaml.Loader)
    BoPt = BoParas(type=config["type"], num_BoOpt=config["num_BoOpt"], random_seed=config["random_seed"], Ex_name=config["Ex_name"])

    BoPt.Opt(**config["cf_opt"])


