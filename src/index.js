const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');
const morgan = require('morgan');
const admin = require('firebase-admin');
const util = require('./logic/utils')

const app = express();
app.use(morgan('combined'));
const port = process.env.PORT || 3000;

// Lista de orígenes permitidos
const allowedOrigins = [
    'https://healthy-plant-jhoandorado25-0296220fc93781a8b02ae446128d2097ec0.gitlab.io',
    'https://5gl00vv6-5173.brs.devtunnels.ms',
];

// Configuración CORS
const corsOptions = {
    origin: (origin, callback) => {
        if (!origin || allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
};

//app.use(cors(corsOptions));

app.use(cors());


const serviceAccount = process.env.SAK || require('/etc/secrets/serviceAccountKey.json');

admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
     storageBucket: 'disease-pest-banana.appspot.com'
});

const bucket = admin.storage().bucket();

const db = admin.firestore();


// Configura multer para recibir archivos en la memoria
const upload = multer({ storage: multer.memoryStorage() });

// Carga el modelo de TensorFlow.js
let modelpseudo;
let modelfruit;
let modelleaf;

try {
    (async () => {

        const data = await util.getCollection("models", db)

        await setModels(data)
    })();
} catch (error) {
    console.log("Error al cargar el modelo, verificar codigo", error)

}

async function setModels(data) {
    const fruitModel = data.find(model => model.model === 'fruit');
    const pseudoModel = data.find(model => model.model === 'pseudo');
    const leafModel = data.find(model => model.model === 'leaf');


    modelpseudo = await tf.loadLayersModel(pseudoModel.url);
    console.log("modelo pseudo cargado")

    modelfruit = await tf.loadLayersModel(leafModel.url);
    console.log("modelo leaf cargado")

    modelleaf = await tf.loadLayersModel(fruitModel.url);
    console.log("modelo fruit cargado")
}


// Ruta para recibir la imagen y hacer la predicción
app.post('/predict/:type', upload.single('image'), async (req, res) => {
    const type = req.params.type;
    if (!req.file) {
        return res.status(400).send('No file uploaded.');
    }
    try {
        // Convierte el buffer de la imagen a un formato que Jimp pueda leer
        const imageBuffer = req.file.buffer;

        const data = await util.processImage(imageBuffer, modelfruit, type)
        console.log(data)

        const fileName = `dataset/${type}/${data[0]}/${Date.now()}-${req.file.originalname}`;
        const file = bucket.file(fileName);
         file.save(imageBuffer, {
            metadata: { contentType: req.file.mimetype }
        });
        // Obtener la URL pública del archivo subido
        //const publicUrl = `https://storage.googleapis.com/${bucket.name}/${fileName}`;
        res.json({
            predicted_class: data[0],
            probability: data[1]
        });
    } catch (error) {
        console.error(error);
        res.status(500).send('Error making prediction.');
    }
});



app.get('/data/:collection', async (req, res) => {
    const collectionName = req.params.collection;
    try {
        const data = await util.getCollection(collectionName, db)

        res.json(data);
    } catch (error) {
        console.error('Error getting documents:', error);
        res.status(500).send('Error getting documents');
    }
});


// Ruta de prueba para CORS
app.get('/test-cors', (req, res) => {
    res.json({ message: 'CORS is working!' });
});


app.get('/changes', async (req, res) => {
    const data = await util.getCollection("models", db)

    await setModels(data)

    res.status(200).send({ message: 'cambios realziados' });
});




app.listen(port, () => {
    console.log(`Server listening`);
});
