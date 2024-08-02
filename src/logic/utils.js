const Jimp = require('jimp');
const tf = require('@tensorflow/tfjs-node');


const classNamesMap = {
    leaf: {
        0: 'sana',
        1: 'moko',
        2: 'sigatoka',
    },
    fruit: {
        0: 'sana',
        1: 'moko',
        2: 'escarabajoCicatrizante',
    },
    pseudo: {
        0: 'sana',
        1: 'picudo',
        2: 'sana',
    }
};

async function processImage(imageBuffer, model, type) {

    let classNames = classNamesMap[type] || {};


    const img = await Jimp.read(imageBuffer);

    // Redimensiona la imagen y convierte el buffer a un formato compatible con TensorFlow.js
    img.resize(250, 250); // Ajusta el tamaño según el modelo

    const resizedImageBuffer = await img.getBufferAsync(Jimp.MIME_PNG);
    // Convierte el buffer de la imagen a un tensor
    const imgTensor = tf.node.decodeImage(resizedImageBuffer, 3); // 3 para RGB

    // Normaliza la imagen
    const normalizedImgTensor = imgTensor.div(tf.scalar(255.0)).expandDims(0);

    // Realizar la predicción
    const predictions = model.predict(normalizedImgTensor);

    const predictionArray = predictions.arraySync();

    const predictedClassIndex = predictionArray[0].indexOf(Math.max(...predictionArray[0]));

    const predictedClassName = classNames[predictedClassIndex];

    return [predictedClassName, predictionArray[0][predictedClassIndex]]

}

async function getCollection(collectionName, db) {

    const snapshot = await db.collection(collectionName).get();
    if (snapshot.empty) {
        console.log("No data found.")
    }
    const data = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
    }));

    return data
}
module.exports = { processImage ,getCollection};