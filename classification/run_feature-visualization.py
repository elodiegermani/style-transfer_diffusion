from utils import feature_visualization
from utils.datasets import ClassifDataset

def main(config):

    dataset_file = f'{config.data_dir}/{config.mode}-{config.dataset}.csv'

    dataset = ClassifDataset(
        dataset_file, 
        config.labels)

        for i, classes in enumerate(dataset.label_list):
            print(classes)
            p=True
            feature_visualization.visualize_features(config.param_file, dataset, classe=classes, 
                                                     classe_name=dataset.label_list[i], types='right-hand', print_title=p)


if __name__ == '__main__':
    
    config = {
        data_dir : '../data',
        dataset : 'dataset_rh_4classes-jeanzay',
        mode : 'test',
        labels : 'pipelines'

    }

    main(config)