const a = document.querySelector('a');

const data = 'Nazir Knuckles at fat meet';

// Create a Blob (Binary Large Object)
const blob = new Blob([data], { type: 'text/plain' });
console.log(data.length);
console.log(blob);

// Create a URL for data in Blob
// const url = URL.createObjectURL(blob);
// console.log(url);


// Set as href
// a.href = url;
// a.download = 'help.csv'



const users = [
    { name: 'Alice', age: 25 },
    { name: 'Bob', age: 30 },
    { name: 'Charlie', age: 35 }
];
console.table(users);


console.time('Nazir Knuckles')


console.timeEnd('Nazir Knuckles')
