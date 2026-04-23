# GiosGL
## GiosGL Offical Git repo

### ---------------------- DOCUMENTATION & EXAMPLES ----------------------------

GiosGL is a one-stop high-end WebGL2 utilities library for interactive graphics
applications, 4K/UHD demos, scientific visualization, and GPGPU experiments.

**Initialization:**
    const engine = new GiosGL(document.getElementById('myCanvas'));

**Shader Pipeline:**
    engine.createProgram('main', GiosGL.exampleVertexShader(), GiosGL.exampleFragmentShader());
    engine.useProgram('main');

**Geometry:**
    let quad = GiosGL.makeQuad();
    engine.createVBO('quad', quad);
    engine.createVAO('quad', [{ vbo: 'quad', index: 0, size: 2 }]);

**Textures:**
    engine.createTexture2D('img1', document.getElementById('myImageElement'));
    engine.bindTexture('img1', 0);

**Uniforms:**
    engine.setUniform1f('time', performance.now() * 0.001);

**Render Loop:**
    engine.run((time, frame) => {
        engine.useProgram('main');
        engine.setUniform1f('time', time);
        engine.bindVAO('quad');
        engine.drawArrays(engine.context.TRIANGLE_STRIP, 4, 0);
        engine.unbindVAO();
    });

**Framebuffer/PostFX:**
    engine.createFramebuffer('offscreen', 1024, 1024);
    engine.renderToFramebuffer('offscreen', () => { ... });

333----------------------------------------------------------------------------
