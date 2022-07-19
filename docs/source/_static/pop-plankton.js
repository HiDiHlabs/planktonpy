function escape_mouse(event) {
    img = document.getElementById('pop-plankton');
    console.log(document.location);

    var hpos = -(50 + Math.floor(Math.random() * 10)) + 'px';
    var vpos = (Math.floor(Math.random() * 80) + 10) + '%';
    var rotation = Math.random() * 0.2 - 0.1 + (0.75);

    console.log(img, '' + vpos);

    img.style.right = "-200px";

    setTimeout(function () {
        img.style.bottom = '' + vpos;
        img.style.transform = "rotate(" + rotation * 360 + "deg)";
        img.style.right = '' + hpos;
    }, 700);
}

var rotation = Math.random() * 0.1 - 0.05 + (0.75);
var hpos = 'right:-' + (50 + Math.floor(Math.random() * 10)) + 'px;';
var vpos = 'bottom:' + (Math.floor(Math.random() * 80) + 10) + '%;';

var img = document.createElement("img");

//"../_images/plankton-only.svg";
if ((document.location.pathname.split('/')).includes("rst")){
    img.src = "../../_images/plankton-only.svg";
}
else if ((document.location.pathname.split('/')).slice(-1)==""){
    img.src = "_images/plankton-only.svg";
}
else{
    img.src = "../_images/plankton-only.svg";
}

img.id = "pop-plankton";
img.onmouseenter = escape_mouse;
img.width = 100;
img.height = 100;
img.style = "transform:rotate(" + rotation * 360 + "deg);position:fixed;z-score:5   ;" + vpos + hpos;//#"+Math.random()*100+"vw;";

document.addEventListener("DOMContentLoaded", function (event) {
    console.log(img);
    var canvas = document.getElementsByClassName('wy-nav-content-wrap')[0];
    canvas.appendChild(img);
});
