const session = require('express-session');
const express = require('express');
const path = require('path');
const app = express();
const PORT = 5000;

app.use(express.static(path.join(__dirname, '../client')))
app.use(express.json());
app.use(session({
    secret: 'secret-key',
    resave: false,
    saveUninitialized: false
}))

app.get('/api', (req, res) => {
    if (req.session.viewCount == 1 || typeof req.session.viewCount === 'string') req.session.viewCount = 'https://google.com'
    else {
        if (req.session.viewCount) req.session.viewCount += 1;
        else req.session.viewCount = 1;
    }
    res.json({ Naz: req.session.viewCount });
})

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../client/index.html'), (error) => {
        if (error) {
            console.error('Error sending file:', error);
            res.status(error.status).end();
        }
    })
})


app.listen(PORT, () => {
    console.log('listening on port: ', PORT)
})