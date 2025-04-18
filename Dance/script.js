const dancer = document.querySelector('.dancer');

document.body.addEventListener('click', () => {
    dancer.style.backgroundColor = '#' + Math.floor(Math.random()*16777215).toString(16);
});

setInterval(() => {
    dancer.style.transform = `translateY(${Math.random() * 100}px) rotate(${Math.random() * 360}deg)`;
}, 1000);
