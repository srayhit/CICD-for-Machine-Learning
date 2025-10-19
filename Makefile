install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.Model
	cat ./Results/metrics.txt >> report.md 

	echo '\n## Confusion Matrix Plot' >> report.md 
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md 


	cml comment create report.md 

update-branch:
	git config --global user.name $("srayhit")
	git config --global user.email $("sambarta1202@outlook.com")
	git commit -am "Update with new results"
	git push --force origin HEAD:update