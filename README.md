# bird_classification
Classification of birds with their voice

Files discription:
1. make_file.py contains the python code for creating the birds directories and a csv file corresponding to every bird that contains respective details of that bird.
2. audio_cut.py contains the python script for cutting the audio for each bird. The trimmed audio might contain the voice of some other birds also. The number of audio files for a bird may vary as per our data.
3. spectrogram.py contains the python script for creating spectrograms from the trimmed audio files of birds obtained from audio_cut.py.
4. The train_test.py file contains the python code for manually splitting the data for training and testing purposes. As per our data the number of audio files of any bird may vary from 1 to 1000. So for a bird file containing less than 5 audio files, I used it for training purpose only. Because testing at that less data is not worth it, will give wrong answer mostly. And for bird directory containg more than 5 audio files, I split the data as 20:80 for testing and training purpose. All files been saved with a unique identity.
5.  DL_model.ipynb file contains the training and testing part for our model. I used google colab for GPU support. The 4th cell of ipynb file contains the training and validating part of our data followed by the testing data. The accuracy for our testing data was 87% for 20 epoch. Was increased for more epoch. And the accuracy for training data was 92% which signifies that our model was not overfitted.
