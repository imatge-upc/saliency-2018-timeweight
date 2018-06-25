# saliency-2018-timeweight
The Importance of Time in Visual Attention Models

# Abstract

Predicting visual attention is a very active field in the computer vision community. Visual attention is a mechanism of the visual system that can select relevant areas within a scene. Models for saliency prediction are intended to automatically predict which regions are likely to be attended by a human observer. Traditionally, ground truth saliency maps are built using only the spatial position of the fixation points, being these fxation points the locations where an observer fixates the gaze when viewing a scene. In this work we explore encoding the temporal information as well, and assess it in the application of prediction saliency maps with deep neural networks. It has been observed that the later fixations in a scanpath are usually selected randomly during visualization, specially in those images with few regions of interest. Therefore, computer vision models have difficulties learning to predict them. In this work, we explore a temporal weighting over the saliency maps to better cope with this random behaviour. The newly proposed saliency representation assigns different weights depending on the position in the sequence of gaze fixations, giving more importance to early timesteps than later ones. We used this maps to train MLNet, a state of the art for predicting saliency maps. MLNet predictions were evaluated and compared to the results obtained when the model has been trained using traditional saliency maps. Finally, we show how the temporally weighted saliency maps brought some improvement when used to weight the visual features in an image retrieval task.

# Bachelor thesis

This repository contains the models and data used in the bachelor thesis of Marta Coll-Pol presented at the Universitat Politecnica de Catalunya (UPC) on the 25th June 2018. This work was co-advised by Kevin McGuinness from Dublin City University and Xavier Giro-i-Nieto from UPC.

Coll-Pol M. [The Importance of Time in Visual Attention Models](https://github.com/imatge-upc/saliency-2018-timeweight/raw/master/MartaColl-2018-slides.pdf). Universitat Politecnica de Catalunya 2018. 

You can find the slides both on [Slideshare](https://www.slideshare.net/xavigiro/the-importance-of-time-in-visual-attention-models) and [PDF](https://github.com/imatge-upc/saliency-2018-timeweight/raw/master/MartaColl-2018-slides.pdf).

![Marta Coll Pol](https://github.com/imatge-upc/saliency-2018-timeweight/blob/master/MartaColl-2018-BSc.jpg?raw=true)

![Marta Coll Pol and Xavier Giro-i-Nieto](https://github.com/imatge-upc/saliency-2018-timeweight/blob/master/MartaColl-2018-BSc2.jpg?raw=true)
