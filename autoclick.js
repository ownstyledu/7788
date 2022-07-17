// ==UserScript==
// @name         Autoclick
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  继续教育系统一看到底
// @author       ddp
// @match        https://m.mynj.cn:11188/zxpx/tec/play/*
// @icon         https://www.google.com/s2/favicons?sz=64&domain=mynj.cn
// @grant        unsafeWindow
// ==/UserScript==


(function() {
    setInterval(function () {
        if($('div.dialog-button.messager-button > a').length>0){
            $('div.dialog-button.messager-button > a').click();
            if($("#player-container-id > div.vjs-control-bar > button.vjs-play-control.vjs-control.vjs-button.vjs-paused").length>0){
               $("#player-container-id > div.vjs-control-bar > button.vjs-play-control.vjs-control.vjs-button.vjs-paused").click();
               console.log('sucess')
            }
        }
    }, 100);
})();