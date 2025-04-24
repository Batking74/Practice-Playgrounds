// Importing Packages
const express = require('express');
const app = express();
const PORT = 5000;

app.get('/', (req, res) => {
    res.json({
        Name: 'Nazir',
        Age: '20',
        Birthday: '03-31-2005',
        Hobbies: ['Coding', 'Playing Video Games']
    });
})


app.listen(PORT, () => console.log(`Listening on Port ${PORT}`));