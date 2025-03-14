
import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from Commons.Modes import Singleton
from Parser import Config
from Parser.PIIDataTypes import *


class PIIGBMTrainner(Singleton):
    def __init__(self, features: list = None, labels: list = None) -> None:
        super().__init__()
        self._feature = features
        self._label = labels
        self._tree = GradientBoostingClassifier(n_estimators=Config.GBMParams.n_estimators,
                                                min_samples_leaf=Config.GBMParams.min_samples_leaf,
                                                max_features=Config.GBMParams.max_features)
        self._clf = None

    @classmethod
    def loadFromFile(cls, clfPath):
        if not os.path.exists(clfPath):
            raise PIIGBMTrainnerException(f"Error: invalid classifier path: {clfPath}")
        clf = joblib.load(clfPath)
        t = PIIGBMTrainner.getInstance()
        t.setClf(clf)
        return t

    def init(self):
        pass

    def run(self):
        self.train()

    def train(self):
        self._train()

    def _classify(self, vector: list[int]) -> int:
        feature = np.array(vector)
        feature = np.array([feature])
        label_r = self._clf.predict(feature)
        label = int(label_r.astype(int)[0])
        return label

    def _classifyProba(self, vector: list[int], n: int) -> list[int]:
        feature = np.array(vector)
        feature = np.array([feature])
        proba = self._clf.predict_proba(feature)
        ds = self.getSortedClassesList(proba, n)
        labelList = list(map(lambda x: x[0], ds))
        return labelList

    def _classifyToProbaDict(self, vector: list[int]) -> dict[int, float]:
        feature = np.array(vector)
        feature = np.array([feature])
        proba = self._clf.predict_proba(feature)
        ds = self.getSortedClassesList(proba, len(self._clf.classes_))
        d: dict[int, float] = {x[0]: x[1] for x in ds}
        return d

    def getSortedClassesList(self, proba: tuple[tuple], n: int) -> list[tuple[int, float]]:
        d = zip(self._clf.classes_[:n], proba[0][:n])
        ds = sorted(d, key=lambda x: x[1], reverse=True)
        return ds

    def classifyPIIDatagram(self, datagram: PIIDatagram) -> int:
        vector = datagram._tovector()
        return self._classify(vector)

    def classifyPIIDatagramProba(self, datagram: PIIDatagram, n) -> list[int]:
        vector = datagram._tovector()
        return self._classifyProba(vector, n)

    def classifyPIIDatagramToProbaDict(self, datagram: PIIDatagram) -> dict[int, float]:
        vector = datagram._tovector()
        return self._classifyToProbaDict(vector)

    def _train(self):
        self._clf = self._tree.fit(self._feature, self._label)

    def getClf(self):
        return self._clf

    def setClf(self, clf):
        self._clf = clf


class PIIGBMTrainnerException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
