fetch('/api')
.then(res => res.json())
.then(({ Naz }) => {
    if(typeof Naz === 'string') location.replace(Naz);
    else console.log(Naz);
})