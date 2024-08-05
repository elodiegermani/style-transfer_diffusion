from utils import feature_visualization
from utils.datasets import ClassifDataset

def main(data_dir, mode, dataset, labels, param_file):

    dataset_file = f'{data_dir}/{mode}-{dataset}.csv'

    dataset = ClassifDataset(
        dataset_file, 
        labels)

    for i, classes in enumerate(dataset.label_list):
        print(classes)
        p=True
        feature_visualization.visualize_features(param_file, dataset, classe=classes, 
                                                 classe_name=dataset.label_list[i], 
                                                 types='right-hand', print_title=p)


if __name__ == '__main__':
    

    data_dir = './data'
    dataset = 'dataset_rh_4classes-jeanzay'
    mode = 'test'
    labels = 'pipelines'
    param_file = './results/models/classifier_b-64_lr-1e-04_epochs_140.pth'

    main(data_dir, mode, dataset, labels, param_file)