var bratLocation = 'static/js';
head.js(
    // External libraries
    bratLocation + '/client/lib/jquery-2.1.4.min.js',
    bratLocation + '/client/lib/jquery.svg-1.5.0.min.js',
    bratLocation + '/client/lib/jquery.svgdom-1.5.0.min.js',

    // brat helper modules
    bratLocation + '/client/src/configuration.js',
    bratLocation + '/client/src/util.js',
    bratLocation + '/client/src/annotation_log.js',
    bratLocation + '/client/lib/webfont.js',

    // brat modules
    bratLocation + '/client/src/dispatcher.js',
    bratLocation + '/client/src/url_monitor.js',
    bratLocation + '/client/src/visualizer.js',

    // Stanza parse viewer
    './stanza-parseviewer.js'
);

var fontsLocation = 'static/fonts'
var webFontURLs = [
    fontsLocation + '/Astloch-Bold.ttf',
    fontsLocation + '/PT_Sans-Caption-Web-Regular.ttf',
    fontsLocation + '/Liberation_Sans-Regular.ttf'
];
