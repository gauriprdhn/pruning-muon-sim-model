import datetime
import sys

from nn_logging import getLogger
logger = getLogger()

from nn_models import save_my_model

def train_model(model, x, y, model_name='model', save_model = False, batch_size=None, epochs=1, verbose=False, callbacks=None,
                validation_split=0., shuffle=True, class_weight=None, sample_weight=None):
  start_time = datetime.datetime.now()
  logger.info('Begin training ...')

  history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                      validation_split=validation_split, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight)

  logger.info('Done training. Time elapsed: {0} sec'.format(str(datetime.datetime.now() - start_time)))
  if save_model:
      save_my_model(model, name=model_name)

  return history