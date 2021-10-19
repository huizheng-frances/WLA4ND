# -WLA4ND

WLA4ND Dataset is the work of the papper -- "WLA4ND: a Wearable Dataset of Learning Activities for Young Adults with Neurodiversity to Provide Support in Education" is for more in-depth details.
  https://dl.acm.org/doi/10.1145/3441852.3471220


The WLA4ND dataset consists of data collected with a set of wearable sensors, representing learning activities from eight participants with neurodiversity. Among the eight participants, there are three females and five males (according to self-reports), all were alumni from an inclusive postsecondary education program for students with intellectual and developmental disabilities. The participants self-reported their disability types. Regarding their demographic profile, seven participants were White-Caucasian, and one was African-American. All eight participants used their right hand as dominant hand. The average age of the participants was 28.5 years old, ranging from 23 to 31 years old. 

We collected data of learning activities performed by eight young adults with neurodiversity collected from smartwatch sensors. The activities are common learning tasks, including reading, writing, typing, answering follow-up questions, and off-task. The data was collected from five smartwatch built-in sensors, including 13 dimensions of features, being 12 dimensions of features from four motion sensors and one feature from the heart rate sensor. For those four motion sensors that measure vectors in three axes (x, y, z) relative to the smartwatch, we included data from the gyroscope (raw rotations), accelerometer (raw accelerations), linear acceleration (accelerations excluding gravity), and gravity sensors to measure the corresponding acceleration vectors. Heart rate sensors were used to measure the number of beats per minute (bpm) in one vector. For the raw data, around 500 raw data points in total were collected from five sensor sources per second. 

There six labels for dataset:
- 'read'  : reading
- 'writeQ&A'  : writing follow-up Question and Answers
- 'write' : writing
- 'type'  : typing
- 'rest'  : taking a rest
- 'off'   : being off-task


For details:
- The paper "WLA4ND: a Wearable Dataset of Learning Activities for Young Adults with Neurodiversity to Provide Support in Education" is for more in-depth details.
  - https://dl.acm.org/doi/10.1145/3441852.3471220

- The "WLA4ND" dataset folder is for the  dataset.  
  - https://github.com/huizheng-frances/WLA4ND/tree/main/WLA4ND_dataset
  - https://github.com/huizheng-frances/WLA4ND/blob/main/WLA4ND_dataset/ReadMe

- The "modles code" folder is for the classification code. 
  -   https://github.com/huizheng-frances/WLA4ND/tree/main/models_code


To use this dataset or related code, please cite our paper:

Hui Zheng, Pattiya Mahapasuthanon, Yujing Chen, Huzefa Rangwala, Anya S Evmenova, and Vivian Genaro Motti. 2021. WLA4ND: a Wearable Dataset of Learning Activities for Young Adults with Neurodiversity to Provide Support in Education. In The 23rd International ACM SIGACCESS Conference on Computers and Accessibility (ASSETS '21). Association for Computing Machinery, New York, NY, USA, Article 29, 1–15. DOI:https://doi.org/10.1145/3441852.3471220
  - BibTex： https://github.com/huizheng-frances/WLA4ND/blob/main/acm_3441852.3471220.bib

If you have any questions, please contact hzheng5@gmu.edu

