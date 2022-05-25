from .portfolio import Portfolio_Dataset
from .generated import Generated_Dataset
from .generated_no_side import Generated_Dataset_no_side


def load_dataset(dataset_name, data_path, test_path, normal_class,xp_path):
    """Loads the dataset."""

    implemented_datasets = ('portfolio','generated','mine')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name =='portfolio':
        dataset = Portfolio_Dataset(root=data_path,test=test_path,normal_class=normal_class)
    
    if dataset_name =='mine':
        dataset = Generated_Dataset(root=data_path,test=test_path,normal_class=normal_class,xp_path=xp_path) 
        
    if dataset_name =='generated':
        dataset = Generated_Dataset_no_side(root=data_path,test=test_path,normal_class=normal_class,xp_path=xp_path)  
        
        
    return dataset
