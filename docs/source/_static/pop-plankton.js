// docoument.getElementById('kewl').background='r';

// var edge = [0,2,3][Math.floor(Math.random() * 3)];
// var rotation = Math.random() * 0.1 - 0.05 + (edge * 0.25);

// var edges = ['bottom', 'right', 'top', 'right']

// var vpos = edges[edge]+':-'+(50+Math.floor(Math.random()*10))+'px;';
// var hpos = edges[(edge+1)%4]+':'+(Math.floor(Math.random()*70)+5)+'%;';

var edge = [0,2,3][Math.floor(Math.random() * 3)];
var rotation = Math.random() * 0.1 - 0.05 + (0.75);

var vpos = 'right:-'+(50+Math.floor(Math.random()*10))+'px;';
var hpos = 'bottom:'+(Math.floor(Math.random()*80)+10)+'%;';

// console.log(vpos,hpos)

var img = document.createElement("img");
img.src = "../_images/plankton-only.svg";
img.id = "pop-plankton";
img.width = 100;
img.height = 100;
img.style = "transform:rotate(" + rotation * 360 + "deg);position:fixed;z-score:5   ;"+vpos+hpos;//#"+Math.random()*100+"vw;";


document.addEventListener("DOMContentLoaded", function (event) {
    console.log(img);

    var canvas = document.getElementsByClassName('wy-nav-content-wrap')[0];
    canvas.appendChild(img);
});
