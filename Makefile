#--------------------------------------
# Targets to run the model pipeline
#--------------------------------------
.PHONY: download

# Download the data
download:
	python3 -m src.data.data_loader

# Preprocess the data
preprocess:
	python3 -m src.preprocess.build_features

# Visualize the data
visualize:
	python3 -m src.visualize.explore_and_visualize

#---------------------------------------------------
# Cleaning folders
#---------------------------------------------------
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Delete all data
clean-data:
	rm -rf data/raw/*
	rm -rf data/interim/*
	rm -rf data/processed/*

# Delete all models, metrics, and visualizations
clean-results:
	rm -rf models/*
	rm -rf results/*
	rm -rf reports/figures/*

# Delete all
clean-all: clean clean-data clean-results