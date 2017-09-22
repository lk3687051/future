import datetime
import logging
import time

today = datetime.date()
logging.debug("Daily task to predict all at " + str(today))

for dataset in datasets:
    logging.debug("Start update dataset " + dataset.name)
    start = time.time()
    dataset.update()
    end = time.time()
    logging.debug("End update dataset " + dataset.name + "Used %d seconds" % (end - start))

for feature in features:
    feature.process()

for model in models:
    logging.debug("Start update dataset " + dataset.name)
    start = time.time()
    model.predict()
    end = time.time()
    logging.debug("End update dataset " + dataset.name + "Used %d seconds" % (end - start))
