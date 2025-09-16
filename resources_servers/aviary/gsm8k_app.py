from app import AviaryResourcesServer
from aviary.envs.gsm8k import CalculatorEnv, GSM8kDataset, GSM8kDatasetSplit
from pydantic import Field


class GSM8kResourcesServer(AviaryResourcesServer[CalculatorEnv, GSM8kDataset]):
    dataset: GSM8kDataset = Field(default_factory=lambda: GSM8kDataset(GSM8kDatasetSplit.train))


if __name__ == "__main__":
    GSM8kResourcesServer.run_webserver()
