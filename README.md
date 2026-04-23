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

----------------------------------------------------------------------------


# Release:
## GiosGL v1.0.0

I am thrilled to announce the stable v1.0.0 release of GiosGL, the definitive WebGL2 wrapper designed explicitly for 4K / 60FPS performance in the browser.This release graduates the engine from a lightweight wrapper to a full-fledged creative computing API.Core Architecture FinalizedNative 4K Handling: The engine now automatically forces a $3840 \times 2160$ internal drawing buffer, keeping visuals pristine on high-density displays.Low-Latency Compositing: Configured context creation with desynchronized: true to bypass the browser's compositor where possible, drastically reducing input lag.Feature HighlightsComprehensive Resource Registries: this.programs, this.textures, this.vaos, and more now safely manage your GPU memory.Hot-Reloadable Shaders: createProgram(id, vs, fs) automatically caches uniform and attribute locations, eliminating costly lookups in the render loop.Advanced Texturing & FBOs: Seamless support for Cubemaps, floating-point textures, and depth renderbuffers for complex GPGPU tasks (like falling sand sims).Hardware Profiling: beginTimerQuery and resolveTimerQuery allow developers to measure exact GPU execution times using EXT_disjoint_timer_query_webgl2.Built-in Math & Events: Zero-dependency matrix math (mat4Perspective, mat4LookAt) and normalized pointer/touch event bindings.Installation:Available via unpkg: https://unpkg.com/glos-gl@1.0.0/GLGiosGL.js
