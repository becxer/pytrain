#
#  HOW TO UPLOAD PROJECT TO PYPI REPOSITORY
#
#  $ python setup.py sdist upload -r pypi
#

from distutils.core import setup

setup(
        name='pytrain',
        version='0.0.12',
        packages = [
            'pytrain',
            'pytrain.lib',
            'pytrain.KNN',
            'pytrain.LinearRegression', 
            'pytrain.LogisticRegression',
            'pytrain.GaussianNaiveBayes', 
            'pytrain.NaiveBayes',
            'pytrain.DecisionTreeID3', 
            'pytrain.Kmeans',
            'pytrain.DBSCAN', 
            'pytrain.Apriori',
            'pytrain.HierarchicalClustering',
            'pytrain.HMM',
            'pytrain.CRF',
            'pytrain.NeuralNetwork',
            'pytrain.SVM'
            ],
        include_package_data=True,
        package_data={'':['*.eng']},
        author ='becxer',
        author_email='becxer87@gmail.com',
        url = 'https://github.com/becxer/pytrain',
        description ='Machinelearning library for python',
        long_description ='Machinelearning library for python',
        license='MIT',
        install_requires=['numpy'],
        classifiers=[
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python :: 2.7',
        ]
)

