from config.train import TrainConfig
from src.utils.helper import run
from src.utils.trainEUC import train1

if __name__ == '__main__':
    config = TrainConfig()
    run(train1, config)
