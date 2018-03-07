#!/usr/bin/env python
import subprocess

#compares runtime of FFT and PCA KNN on that dataset

datasets = ["50words", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF", "ChlorineConcentration", "CinC", "Coffee", "Computers", "Cricket", "Cricket", "Cricket", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FISH", "FordA", "FordB", "Gun", "Ham", "HandOutlines", "Haptics", "Herring", "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lighting2", "Lighting7", "MALLAT", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFatalECG", "NonInvasiveFatalECG", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface", "SonyAIBORobotSurfaceII", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols", "synthetic", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "Two", "UWaveGestureLibraryAll", "uWaveGestureLibrary", "uWaveGestureLibrary", "uWaveGestureLibrary", "wafer", "Wine", "WordsSynonyms", "Worms", "WormsTwoClass", "yoga", "MNIST_all"]
datasets += ["bad", "medium", "good", "bad_big", "medium_big", "good_big"]


datasets = ['labeled_bad_sinusoids_1e4_1e3_trunc']
#datasets = ['StarLightCurves']

proc = "java -Xms16g -Xmx50g ${JAVA_OPTS} -cp 'target/dependency/*:target/classes/' experiments.KNNRuntimeComparison %s"
comp = "mvn compile"
package = "mvn package"
subprocess.call(comp, shell=True)
subprocess.call(package, shell=True)

for dataset in datasets:
    subprocess.call(proc % (dataset), shell=True)


