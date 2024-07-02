from utils import preprocessing

if __name__ == '__main__':
	data_dir = '/gpfswork/rech/gft/umh25bv/hcp_many_pipelines/group*left-hand*con.nii*'
	output_dir = '/gpfswork/rech/gft/umh25bv/hcp_many_pipelines_preprocess'
	
	preprocessing.preprocessing(data_dir, output_dir)