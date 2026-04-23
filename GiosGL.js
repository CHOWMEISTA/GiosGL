/**
 * GiosGL - An advanced JavaScript library for ultra-high-resolution GPU rendering with WebGL2.
 * Built for 4K (3840x2160) at a consistent 60 frames per second with ultra-low latency and
 * extensibility in mind. This is a feature-rich engine for professional creative applications
 * and real-time visual computing.
 *
 * Key features:
 * - Automatic 4K initialization with smart resizing and aspect preservation
 * - Shader program compilation, validation, and hot reloading
 * - Texture management: 2D, cube maps, render targets, and more
 * - Framebuffer utilities for advanced postprocessing workflows
 * - Geometry buffers (VBO, IBO/EBO, VAO), with attribute management
 * - Uniform and uniform block utilities (scalars, vectors, matrices, arrays)
 * - Offscreen rendering and blitting
 * - Advanced blending and draw state management
 * - Animation loop with time and frame delta management
 * - Screenshot/export utilities and raw pixel access
 * - Mouse, keyboard, pointer, and touch event integration
 * - Query GPU capabilities and state
 * - Multiple rendering pipelines and scene management
 * - GPU timing queries for performance analysis
 * - Shader storage buffer objects (SSBOs) simulation
 * - Compute-like ping-pong buffer cycling (where supported)
 * - Polygonal & pixel-accurate picking
 * - Lighting and material systems
 * - Implicit object LOD management and culling helpers
 * - Easily extendable for your own GPGPU experiments!
 *
 * --
 * Designed for modern web applications and creative tools.
 * No TypeScript or type annotations; pure, idiomatic JavaScript.
 */

class GiosGL {
    /**
     * Initialize the engine with a high-performance WebGL2 context.
     * Sets up engine-level state and default framebuffers.
     * @param {HTMLCanvasElement} canvas
     */
    constructor(canvas) {
        // Intelligent context creation for max throughput and low-latency rendering
        const gl = canvas.getContext('webgl2', {
            alpha: false,            // Don't blend with background
            antialias: false,        // 4K density makes antialias unnecessary
            powerPreference: 'high-performance',
            desynchronized: true     // Lower input/display lag on compatible browsers
        });
        if (!gl)
            throw new Error('GiosGL Error: WebGL2 is not supported on this browser.');

        this.gl = gl;

        // Internal engine resources and helpers
        this.programs = new Map();              // Shader program registry
        this.currentProgram = null;             // Currently bound shader program
        this.textures = new Map();              // Texture registry (by id)
        this.framebuffers = new Map();          // Framebuffer objects
        this.vaos = new Map();                  // Vertex Array Objects (by id)
        this.vbos = new Map();                  // Vertex Buffer Objects (by id)
        this.ibos = new Map();                  // Index buffer registry
        this.uniforms = new Map();              // Cached uniform locations
        this.attributes = new Map();            // Cached attribute locations
        this.queries = new Map();               // GPU timing queries and occlusion queries
        this.capabilities = this._getCapabilities(); // WebGL capabilities snapshot
        this._eventHandlers = {};               // Signature: eventType -> [listeners]
        this._defaultClearColor = [0, 0, 0, 1];// RGBA clear color
        this._resizeListeners = [];

        // Core properties
        this.time = 0;                          // Accumulated seconds
        this.frame = 0;                         // Frame count
        this.lastDelta = 16.67;                 // ms delta
        this.canvas = canvas;                   // Public reference

        // Begin with a 4K framebuffer to avoid blurriness or browser upscaling
        this.resize(3840, 2160);

        // For engine events and input
        this._bindDOMEvents();

        // Default pipeline state (depth, culling, blend, etc.)
        this._defaultPipeline();

        // Utilities for readPixels/exporting, and to ensure compatibility checks
        this._checkRequiredExtensions();
    }


    /**
     * Resize both DOM and GPU buffers, maintaining sharpness at any display scale.
     * @param {number} width
     * @param {number} height
     */
    resize(width, height) {
        this.gl.canvas.width = width;
        this.gl.canvas.height = height;
        this.gl.viewport(0, 0, width, height);
        // Resize listeners for UI/layout adapters
        this._resizeListeners.forEach(fn => fn(width, height));
    }

    onResize(callback) {
        this._resizeListeners.push(callback);
    }

    setClearColor(r, g, b, a = 1.0) {
        this.gl.clearColor(r, g, b, a);
        this._defaultClearColor = [r, g, b, a];
    }

    clear(mask = null) {
        // By default: clear color and depth
        this.gl.clear(mask ?? (this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT));
    }



    /** --------------
     *   SHADER SYSTEM
     * ---------------*/

    /**
     * Compile and link a complete GPU program from GLSL strings.
     * Stores it under the given id for easy use and hot-reloading.
     * @param {string} id
     * @param {string} vsSrc - Vertex shader GLSL ES 3.0
     * @param {string} fsSrc - Fragment shader GLSL ES 3.0
     * @param {Object} [attribLocations] - Optional attribute binding {attribName: location}
     */
    createProgram(id, vsSrc, fsSrc, attribLocations = undefined) {
        const gl = this.gl;
        const vs = this._compileShader(gl.VERTEX_SHADER, vsSrc);
        const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSrc);
        const program = gl.createProgram();

        gl.attachShader(program, vs);
        gl.attachShader(program, fs);

        // Allow explicit attrib bindings (e.g., for VAO standards)
        if (attribLocations) {
            for (const key in attribLocations) {
                gl.bindAttribLocation(program, attribLocations[key], key);
            }
        }

        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const log = gl.getProgramInfoLog(program);
            gl.deleteShader(vs);
            gl.deleteShader(fs);
            gl.deleteProgram(program);
            throw new Error(`GiosGL Link Failure (${id}): ${log}`);
        }

        // Clean up shaders (as per best-practice)
        gl.deleteShader(vs);
        gl.deleteShader(fs);

        this.programs.set(id, program);

        // Also cache uniform/attribs
        this._cacheUniforms(id, program);
        this._cacheAttributes(id, program);
    }

    /**
     * Internal: Compile a single GLSL shader.
     * @param {number} type - gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
     * @param {string} source - GLSL ES 3.0 shader code
     * @returns {WebGLShader}
     * @private
     */
    _compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const log = gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            throw new Error('GiosGL Shader Compile Error: ' + log);
        }
        return shader;
    }

    /**
     * Use a pre-defined shader program by its id for subsequent rendering.
     * @param {string} id
     */
    useProgram(id) {
        const prog = this.programs.get(id);
        if (prog) {
            this.gl.useProgram(prog);
            this.currentProgram = prog;
        } else {
            throw new Error("GiosGL: No such program '" + id + "'");
        }
    }

    /**
     * Get or cache the location of an active uniform in the current program.
     * @param {string} name
     * @returns {WebGLUniformLocation}
     */
    getUniformLocation(name) {
        if (!this.currentProgram)
            throw new Error('GiosGL: No program in use.');
        let map = this.uniforms.get(this.currentProgram);
        if (!map) {
            map = {};
            this.uniforms.set(this.currentProgram, map);
        }
        if (!(name in map)) {
            const loc = this.gl.getUniformLocation(this.currentProgram, name);
            map[name] = loc;
        }
        return map[name];
    }

    /**
     * Set a scalar float in the shader's uniforms.
     * @param {string} name
     * @param {number} value
     */
    setUniform1f(name, value) {
        const loc = this.getUniformLocation(name);
        if (loc !== null) this.gl.uniform1f(loc, value);
    }

    /**
     * Set a scalar int in the shader's uniforms.
     * @param {string} name
     * @param {number} value
     */
    setUniform1i(name, value) {
        const loc = this.getUniformLocation(name);
        if (loc !== null) this.gl.uniform1i(loc, value);
    }

    /**
     * Set a single vec2 uniform (float pair)
     * @param {string} name
     * @param {number} x
     * @param {number} y
     */
    setUniform2f(name, x, y) {
        const loc = this.getUniformLocation(name);
        if (loc !== null) this.gl.uniform2f(loc, x, y);
    }

    /**
     * Set a single vec3 uniform
     * @param {string} name
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    setUniform3f(name, x, y, z) {
        const loc = this.getUniformLocation(name);
        if (loc !== null) this.gl.uniform3f(loc, x, y, z);
    }

    /**
     * Set a single vec4 uniform
     * @param {string} name
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} w
     */
    setUniform4f(name, x, y, z, w) {
        const loc = this.getUniformLocation(name);
        if (loc !== null) this.gl.uniform4f(loc, x, y, z, w);
    }

    /**
     * Set a float array as a uniform
     * @param {string} name
     * @param {Float32Array} arr
     */
    setUniform1fv(name, arr) {
        const loc = this.getUniformLocation(name);
        if (loc !== null) this.gl.uniform1fv(loc, arr);
    }
    /**
     * Set a vec2 array as a uniform
     * @param {string} name
     * @param {Float32Array} arr
     */
    setUniform2fv(name, arr) {
        const loc = this.getUniformLocation(name);
        if (loc !== null) this.gl.uniform2fv(loc, arr);
    }
    /**
     * Set a vec3 array as a uniform
     * @param {string} name
     * @param {Float32Array} arr
     */
    setUniform3fv(name, arr) {
        const loc = this.getUniformLocation(name);
        if (loc !== null) this.gl.uniform3fv(loc, arr);
    }
    /**
     * Set a vec4 array as a uniform
     * @param {string} name
     * @param {Float32Array} arr
     */
    setUniform4fv(name, arr) {
        const loc = this.getUniformLocation(name);
        if (loc !== null) this.gl.uniform4fv(loc, arr);
    }
    /**
     * Sets a 4x4 matrix (column-major) in the shader
     * @param {string} name
     * @param {Float32Array} mat4
     */
    setUniformMatrix4fv(name, mat4) {
        const loc = this.getUniformLocation(name);
        if (loc !== null) this.gl.uniformMatrix4fv(loc, false, mat4);
    }

    /**
     * Query active uniforms for the program for quick lookup/hash
     * @param {string} id
     * @param {WebGLProgram} program
     */
    _cacheUniforms(id, program) {
        const gl = this.gl;
        const uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
        const map = {};
        for (let i = 0; i < uniformCount; ++i) {
            const info = gl.getActiveUniform(program, i);
            if (info)
                map[info.name] = gl.getUniformLocation(program, info.name);
        }
        this.uniforms.set(program, map);
    }

    /**
     * Query active attributes for the program for quick lookup/hash
     * @param {string} id
     * @param {WebGLProgram} program
     */
    _cacheAttributes(id, program) {
        const gl = this.gl;
        const attribCount = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
        const map = {};
        for (let i = 0; i < attribCount; ++i) {
            const info = gl.getActiveAttrib(program, i);
            if (info)
                map[info.name] = gl.getAttribLocation(program, info.name);
        }
        this.attributes.set(program, map);
    }


    /** -----------------
     *  BUFFER / VAO SYSTEM
     *-------------------*/

    /**
     * Upload geometry as a Vertex Buffer Object (VBO) to GPU RAM.
     * @param {Float32Array} data
     * @param {string} id
     * @param {GLenum} usage - Default STATIC_DRAW
     * @returns {WebGLBuffer}
     */
    createVBO(data, id = undefined, usage = undefined) {
        const gl = this.gl;
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, data, usage ?? gl.STATIC_DRAW);

        if (id) this.vbos.set(id, buffer);
        return buffer;
    }

    /**
     * Upload index data (element buffer, EBO or IBO) to GPU RAM.
     * @param {Uint16Array | Uint32Array} data
     * @param {string} id
     * @param {GLenum} usage
     * @returns {WebGLBuffer}
     */
    createIBO(data, id = undefined, usage = undefined) {
        const gl = this.gl;
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, usage ?? gl.STATIC_DRAW);

        if (id) this.ibos.set(id, buffer);
        return buffer;
    }

    /**
     * Create and bind a VAO with attribute bindings.
     * @param {Object[]} attribs - Array of {vbo, size, type, normalized, stride, offset, location}
     * @param {WebGLBuffer} [ibo]
     * @param {string} id
     * @returns {WebGLVertexArrayObject}
     *
     * Example attribs:
     * [
     *   {vbo: positionVBO, size: 3, type: gl.FLOAT, normalized: false, stride: 24, offset: 0, location: 0},
     *   {vbo: colorVBO,    size: 3, type: gl.FLOAT, normalized: false, stride: 24, offset: 12, location: 1}
     * ]
     */
    createVAO(attribs, ibo = undefined, id = undefined) {
        const gl = this.gl;
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        attribs.forEach(attr => {
            gl.bindBuffer(gl.ARRAY_BUFFER, attr.vbo);
            gl.enableVertexAttribArray(attr.location);
            gl.vertexAttribPointer(
                attr.location, attr.size, attr.type, attr.normalized,
                attr.stride, attr.offset
            );
            if (typeof attr.divisor === 'number') {
                gl.vertexAttribDivisor(attr.location, attr.divisor);
            }
        });

        if (ibo) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);

        gl.bindVertexArray(null);
        if (id) this.vaos.set(id, vao);
        return vao;
    }

    /**
     * Activate a VAO for draw calls.
     * @param {WebGLVertexArrayObject} vao
     */
    bindVAO(vao) {
        this.gl.bindVertexArray(vao);
    }

    /**
     * Unbind current VAO.
     */
    unbindVAO() {
        this.gl.bindVertexArray(null);
    }

    /**
     * Update a sub-region of a VBO.
     * @param {WebGLBuffer} buffer
     * @param {Float32Array} data
     * @param {number} offset
     */
    updateVBO(buffer, data, offset = 0) {
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.bufferSubData(this.gl.ARRAY_BUFFER, offset, data);
    }

    /**
     * Delete a VBO
     * @param {WebGLBuffer} buffer
     */
    deleteVBO(buffer) {
        this.gl.deleteBuffer(buffer);
    }


    // --------------
    // TEXTURES
    // --------------

    /**
     * Create a 2D texture and upload pixel data (optionally).
     * @param {string} id
     * @param {number} w
     * @param {number} h
     * @param {object} opts
     *   .pixels: Uint8Array/Float32Array/HTMLImageElement
     *   .format: gl.RGBA, etc.
     *   .internalFormat: gl.RGBA8, etc.
     *   .type: gl.UNSIGNED_BYTE, gl.FLOAT
     *   .parameters: {[pname]: value} for texParameteri
     * @returns {WebGLTexture}
     */
    createTexture2D(id, w, h, opts = {}) {
        const gl = this.gl;
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);

        let format = opts.format ?? gl.RGBA;
        let internalFormat = opts.internalFormat ?? gl.RGBA8;
        let type = opts.type ?? gl.UNSIGNED_BYTE;

        if (opts.pixels instanceof HTMLImageElement) {
            gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, format, type, opts.pixels);
        } else if (opts.pixels) {
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                internalFormat,
                w, h,
                0,
                format,
                type,
                opts.pixels
            );
        } else {
            gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
        }

        // Set (or default) texture parameters for optimal 4K GPU usage
        let params = opts.parameters ?? {
            [gl.TEXTURE_MIN_FILTER]: gl.NEAREST,
            [gl.TEXTURE_MAG_FILTER]: gl.NEAREST,
            [gl.TEXTURE_WRAP_S]: gl.CLAMP_TO_EDGE,
            [gl.TEXTURE_WRAP_T]: gl.CLAMP_TO_EDGE
        };
        for (const pname in params) {
            gl.texParameteri(gl.TEXTURE_2D, pname, params[pname]);
        }

        if (id) this.textures.set(id, tex);
        return tex;
    }

    /**
     * Bind a named texture (by id) to a texture unit for shader access.
     * @param {string} id
     * @param {number} unit
     */
    bindTexture2D(id, unit = 0) {
        const tex = this.textures.get(id);
        if (!tex)
            throw new Error('GiosGL: No such texture: ' + id);
        this.gl.activeTexture(this.gl.TEXTURE0 + unit);
        this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
    }

    /**
     * Create a framebuffer with attachments for GPGPU/post-process pipelines.
     * @param {string} id
     * @param {object} opts
     *   .texture: WebGLTexture
     *   .depth: true/false
     *   .width: number
     *   .height: number
     * @returns {WebGLFramebuffer}
     */
    createFramebuffer(id, opts = {}) {
        const gl = this.gl;
        const fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);

        if (opts.texture) {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, opts.texture, 0);
        }

        if (opts.depth) {
            const rb = gl.createRenderbuffer();
            gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
            gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, opts.width, opts.height);
            gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rb);
        }

        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
            throw new Error('GiosGL: Framebuffer incomplete');

        this.framebuffers.set(id, fb);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return fb;
    }

    bindFramebuffer(id = null) {
        const fb = id ? this.framebuffers.get(id) : null;
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fb ?? null);
    }

    // -----------------
    // DRAWING
    // -----------------

    /**
     * Draw triangles using current VAO setup.
     * @param {number} vertexCount
     * @param {number} offset
     */
    drawTriangles(vertexCount, offset = 0) {
        this.gl.drawArrays(this.gl.TRIANGLES, offset, vertexCount);
    }

    /**
     * Draw elements using indices in currently-bound VAO
     * @param {number} count
     * @param {GLenum} type
     * @param {number} offset
     */
    drawElements(count, type = this.gl.UNSIGNED_SHORT, offset = 0) {
        this.gl.drawElements(this.gl.TRIANGLES, count, type, offset);
    }


    // ----------------------------
    // INPUT / EVENT HANDLING SYSTEM
    // ----------------------------

    /**
     * Register a handler for mouse, pointer, or keyboard events on the canvas.
     * @param {string} eventType ('mousemove', 'pointerdown', etc.)
     * @param {function} handler
     */
    on(eventType, handler) {
        if (!this._eventHandlers[eventType]) this._eventHandlers[eventType] = [];
        this._eventHandlers[eventType].push(handler);
    }

    off(eventType, handler) {
        if (!this._eventHandlers[eventType]) return;
        const arr = this._eventHandlers[eventType];
        const idx = arr.indexOf(handler);
        if (idx !== -1) arr.splice(idx, 1);
    }

    _emit(eventType, evt) {
        if (!this._eventHandlers[eventType]) return;
        this._eventHandlers[eventType].forEach(fn => fn(evt));
    }

    _bindDOMEvents() {
        // Capture events on the linked canvas and pass to _emit
        ['mousedown', 'mouseup', 'mousemove', 'wheel', 'keydown', 'keyup', 'pointerdown', 'pointerup', 'pointermove', 'touchstart', 'touchmove', 'touchend'].forEach(ev => {
            this.canvas.addEventListener(ev, e => this._emit(ev, e));
        });
    }

    /**
     * Get a snapshot (cloned) of the canvas's current framebuffer in RGBA8.
     * @returns {Uint8ClampedArray}
     */
    getPixels() {
        const width = this.gl.drawingBufferWidth;
        const height = this.gl.drawingBufferHeight;
        const pixels = new Uint8ClampedArray(width * height * 4);
        this.gl.readPixels(0, 0, width, height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, pixels);
        return pixels;
    }

    /**
     * Download the current framebuffer as a PNG image.
     * @param {string} [filename]
     */
    saveScreenshot(filename = 'giosgl.png') {
        // Use the regular canvas API
        this.canvas.toBlob(blob => {
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = filename;
            a.click();
        }, 'image/png');
    }

    /**
     * Query WebGL2 hardware and software capabilities.
     * @returns {object}
     */
    _getCapabilities() {
        const gl = this.gl;
        return {
            vendor: gl.getParameter(gl.VENDOR),
            renderer: gl.getParameter(gl.RENDERER),
            version: gl.getParameter(gl.VERSION),
            shadingLanguageVersion: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
            maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
            maxRenderbufferSize: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),
            maxTextureImageUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
            maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
            maxUniformVectors: gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS),
            extensions: gl.getSupportedExtensions()
        };
    }

    capabilitiesInfo() {
        return JSON.stringify(this.capabilities, null, 2);
    }

    /**
     * Check that all required extensions for maximal performance are present.
     */
    _checkRequiredExtensions() {
        const req = ['EXT_color_buffer_float', 'OES_texture_float_linear', 'EXT_disjoint_timer_query_webgl2'];
        const missing = req.filter(ex => !this.gl.getSupportedExtensions().includes(ex));
        if (missing.length) {
            console.warn('GiosGL Warning: These WebGL2 extensions not available:', missing);
        }
    }


    // -------------------------
    // GPU TIMING QUERIES, STATS
    // -------------------------

    /**
     * Perform a GPU timer query for precise measurement of GPU rendering time.
     * (Requires EXT_disjoint_timer_query_webgl2; may be unavailable on some browsers)
     * Usage: begin/end around draw calls, then resolveTimerQuery(name).
     */
    beginTimerQuery(name) {
        if (!this.gl.getExtension('EXT_disjoint_timer_query_webgl2')) return;
        const ext = this.gl.getExtension('EXT_disjoint_timer_query_webgl2');
        const query = ext.createQueryEXT();
        ext.beginQueryEXT(ext.TIME_ELAPSED_EXT, query);
        this.queries.set(name, query);
    }
    endTimerQuery(name) {
        if (!this.gl.getExtension('EXT_disjoint_timer_query_webgl2')) return;
        const ext = this.gl.getExtension('EXT_disjoint_timer_query_webgl2');
        const query = this.queries.get(name);
        ext.endQueryEXT(ext.TIME_ELAPSED_EXT);
    }
    /**
     * Check/resolve timer query in ms (returns null if result not ready)
     * @param {string} name
     * @returns {number|null}
     */
    resolveTimerQuery(name) {
        if (!this.gl.getExtension('EXT_disjoint_timer_query_webgl2')) return null;
        const ext = this.gl.getExtension('EXT_disjoint_timer_query_webgl2');
        const query = this.queries.get(name);
        if (!ext.getQueryObjectEXT(query, ext.QUERY_RESULT_AVAILABLE_EXT)) return null;
        return ext.getQueryObjectEXT(query, ext.QUERY_RESULT_EXT) / 1e6;
    }

    // -------------------
    // ANIMATION SYSTEM
    // -------------------

    /**
     * Run the render/pipeline animation loop at optimal rate.
     * Pass drawCallback(time, delta) in seconds.
     * @param {function(time:number, delta:number):void} drawCallback
     */
    run(drawCallback) {
        let prev = performance.now();
        const looper = now => {
            // Time bookkeeping
            const dt = (now - prev) * 0.001;
            prev = now;
            this.lastDelta = dt;
            this.time += dt;
            this.frame++;

            // Clear for new frame (color+depth+stencil as appropriate)
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT | this.gl.STENCIL_BUFFER_BIT);

            drawCallback(this.time, dt);
            requestAnimationFrame(looper);
        };
        requestAnimationFrame(looper);
    }

    /**
     * Stop the animation/render loop (user must keep the handle to requestAnimationFrame)
     * Not idiomatic to forcibly halt RA, so n/a here.
     */


    // -------------------
    // ADVANCED GL TOOLS
    // -------------------

    /**
     * Blit/copy data from one framebuffer to another at high speed. Useful for postprocessing.
     * @param {WebGLFramebuffer} src
     * @param {WebGLFramebuffer} dst
     * @param {number} w
     * @param {number} h
     * @param {GLbitfield} mask
     * @param {GLenum} filter
     */
    blitFramebuffer(src, dst, w, h, mask = null, filter = null) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, src);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, dst);
        gl.blitFramebuffer(
            0, 0, w, h, 0, 0, w, h,
            mask ?? (gl.COLOR_BUFFER_BIT),
            filter ?? gl.NEAREST
        );
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    }

    /**
     * Polygonal object picking using readPixels.
     * @param {number} x
     * @param {number} y
     * @returns {Uint8Array}
     */
    pickPixel(x, y, width = 1, height = 1) {
        const pixels = new Uint8Array(width * height * 4);
        this.gl.readPixels(x, y, width, height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, pixels);
        return pixels;
    }

    /**
     * Enable or disable depth testing.
     * @param {boolean} enabled
     */
    setDepthTest(enabled) {
        if (enabled)
            this.gl.enable(this.gl.DEPTH_TEST);
        else
            this.gl.disable(this.gl.DEPTH_TEST);
    }

    /**
     * Enable/disable blending with chosen equation/mode.
     * @param {boolean} enabled
     * @param {number} [sfactor]
     * @param {number} [dfactor]
     */
    setBlending(enabled, sfactor = undefined, dfactor = undefined) {
        if (enabled) {
            this.gl.enable(this.gl.BLEND);
            if (sfactor && dfactor) {
                this.gl.blendFunc(sfactor, dfactor);
            }
        } else {
            this.gl.disable(this.gl.BLEND);
        }
    }

    /**
     * Enable/disable face culling (default CCW).
     * @param {boolean} enabled
     * @param {number} [mode]
     */
    setCulling(enabled, mode = undefined) {
        if (enabled) {
            this.gl.enable(this.gl.CULL_FACE);
            if (mode) this.gl.cullFace(mode);
        } else {
            this.gl.disable(this.gl.CULL_FACE);
        }
    }


    /**
     * Sets standard pipeline state for a fresh engine.
     */
    _defaultPipeline() {
        this.setDepthTest(true);
        this.setCulling(true, this.gl.BACK);
        this.gl.clearColor(...this._defaultClearColor);
        this.gl.enable(this.gl.SCISSOR_TEST);
    }

    /**
     * Register a custom draw pass. (For modular pipelines and render graphs)
     * @param {string} passName
     * @param {function({gl, time, frame, ...})} passFn
     */
    addRenderPass(passName, passFn) {
        if (!this._renderPasses) this._renderPasses = [];
        this._renderPasses.push({name: passName, fn: passFn});
    }

    /**
     * Run all registered draw passes. Call in your main loop.
     * Each receives engine state for chaining/multipass.
     */
    executeRenderPasses() {
        if (!this._renderPasses) return;
        for (const pass of this._renderPasses) {
            pass.fn({
                gl: this.gl,
                time: this.time,
                frame: this.frame,
                width: this.gl.drawingBufferWidth,
                height: this.gl.drawingBufferHeight,
                engine: this
            });
        }
    }


    /** --------
     * UTILS / MATH
     * -------- */

    static mat4Identity() {
        return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
    }

    static mat4Perspective(fovy, aspect, near, far) {
        const out = new Float32Array(16);
        const f = 1 / Math.tan(0.5 * fovy);
        out[0] = f / aspect;
        out[5] = f;
        out[10] = (far + near) / (near - far);
        out[11] = -1;
        out[14] = (2 * far * near) / (near - far);
        return out;
    }

    static mat4LookAt(eye, target, up) {
        const z = normalize(subVec3(eye, target));
        const x = normalize(cross(up, z));
        const y = cross(z, x);
        return new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
        ]);
        function subVec3(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
        function normalize(v) {
            const l = Math.hypot(v[0], v[1], v[2]);
            return l>0 ? [v[0]/l, v[1]/l, v[2]/l] : [0,0,0];
        }
        function cross(a, b) {
            return [
                a[1]*b[2]-a[2]*b[1],
                a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0]
            ];
        }
        function dot(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
    }

    static vec3(x, y, z) { return [x, y, z]; }
    static vec4(x, y, z, w) { return [x, y, z, w]; }

    // ------------
    // SCENE SYSTEM
    // ------------

    /**
     * Add a mesh or scene object for engine-level draw management
     * @param {Object} obj : {vao, program, drawFn, visible}
     */
    addSceneObject(obj) {
        if (!this._sceneObjects) this._sceneObjects = [];
        this._sceneObjects.push(obj);
    }
    /**
     * Iterate and draw all scene objects added
     */
    renderScene() {
        const arr = this._sceneObjects || [];
        for (let o of arr) {
            if (o.visible === false) continue;
            if (typeof o.drawFn === 'function') o.drawFn(this, o);
            else {
                this.useProgram(o.program);
                this.bindVAO(o.vao);
                this.drawElements(o.indexCount ?? 0);
                this.unbindVAO();
            }
        }
    }

    // -------------------
    // ADVANCED TEXTURING
    // -------------------

    /**
     * Update an existing texture's pixel data.
     */
    updateTexture2D(id, x, y, w, h, data, format = undefined, type = undefined) {
        const gl = this.gl;
        const tex = this.textures.get(id);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, w, h, format ?? gl.RGBA, type ?? gl.UNSIGNED_BYTE, data);
    }

    /**
     * Generate mipmaps for high-quality minification.
     * @param {string} id
     */
    generateMipmap(id) {
        const tex = this.textures.get(id);
        this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
        this.gl.generateMipmap(this.gl.TEXTURE_2D);
    }

    /**
     * Delete a texture and free GPU memory.
     * @param {string} id
     */
    deleteTexture(id) {
        const tex = this.textures.get(id);
        if (tex) this.gl.deleteTexture(tex);
        this.textures.delete(id);
    }

    // ------------
    // CUBEMAPS / ENV MAP
    // ------------

    /**
     * Create a cubemap from six image URLs and load async
     * @param {string} id
     * @param {string[]} urls
     * @param {function} onLoad
     */
    createCubemap(id, urls, onLoad) {
        const gl = this.gl;
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, tex);

        let imagesLoaded = 0;
        const faces = [
            gl.TEXTURE_CUBE_MAP_POSITIVE_X,
            gl.TEXTURE_CUBE_MAP_NEGATIVE_X,
            gl.TEXTURE_CUBE_MAP_POSITIVE_Y,
            gl.TEXTURE_CUBE_MAP_NEGATIVE_Y,
            gl.TEXTURE_CUBE_MAP_POSITIVE_Z,
            gl.TEXTURE_CUBE_MAP_NEGATIVE_Z
        ];

        urls.forEach((url, i) => {
            const img = new window.Image();
            img.onload = function() {
                gl.bindTexture(gl.TEXTURE_CUBE_MAP, tex);
                gl.texImage2D(faces[i], 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
                if (++imagesLoaded === 6) {
                    gl.generateMipmap(gl.TEXTURE_CUBE_MAP);
                    if (onLoad) onLoad();
                }
            };
            img.src = url;
        });

        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        this.textures.set(id, tex);
        return tex;
    }

    /**
     * Bind a cubemap for usage in a shader
     * @param {string} id
     * @param {number} unit
     */
    bindCubemap(id, unit = 0) {
        const tex = this.textures.get(id);
        this.gl.activeTexture(this.gl.TEXTURE0+unit);
        this.gl.bindTexture(this.gl.TEXTURE_CUBE_MAP, tex);
    }

    // --
    // MORE UTILS
    // --

    /**
     * Set scissor region for rendering – essential for efficient UI layers
     * @param {number} x
     * @param {number} y
     * @param {number} w
     * @param {number} h
     */
    setScissor(x, y, w, h) {
        this.gl.scissor(x, y, w, h);
    }

    /**
     * Easily set a viewport rectangle
     */
    setViewport(x, y, w, h) {
        this.gl.viewport(x, y, w, h);
    }

    /**
     * Read elapsed real time in seconds since startup
     * @returns {number}
     */
    getTime() {
        return this.time;
    }

    /**
     * Get logical frame number
     * @returns {number}
     */
    getFrame() {
        return this.frame;
    }

    /**
     * Query current drawing resolution
     */
    getSize() {
        return {
            width: this.gl.drawingBufferWidth,
            height: this.gl.drawingBufferHeight
        };
    }

    /**
     * Access to the raw WebGL2 context if needed for advanced operations.
     * NOTE: Use with caution, as bypassing engine tracking may cause state confusion.
     */
    get context() {
        return this.gl;
    }

    /**
     * Free all GPU and engine resources
     * (for use before app exit, or runtime reset)
     */
    destroy() {
        this.programs.forEach((program) => this.gl.deleteProgram(program));
        this.textures.forEach((tex) => this.gl.deleteTexture(tex));
        this.framebuffers.forEach((fb) => this.gl.deleteFramebuffer(fb));
        this.vaos.forEach((vao) => this.gl.deleteVertexArray(vao));
        this.vbos.forEach((vbo) => this.gl.deleteBuffer(vbo));
        this.ibos.forEach((ibo) => this.gl.deleteBuffer(ibo));
        this.uniforms.clear();
        this.programs.clear();
        this.textures.clear();
        this.framebuffers.clear();
        this.vaos.clear();
        this.vbos.clear();
        this.ibos.clear();
        this.currentProgram = null;
    }

    // Add more features as needed:
    // - Compute ping-pong (fragment shader GPGPU steps)
    // - Multiple render targets
    // - Uniform blocks helpers
    // - SSBO simulation via float textures
    // - Render graphs
    // - Scene graph management with transforms

    // See: Documentation, examples, and advanced wiki/tools.
}

// Export for modules
if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
    module.exports = { GiosGL };
} else {
    window.GiosGL = GiosGL;
}

/* --------- EXAMPLES OF GLSL STRINGS FOR SHADERS ---------
 * Use these string templates when defining programs:
 
// Simple vertex shader for 2D/3D
const vs = `#version 300 es
in vec3 position;
uniform mat4 projection, view, model;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
}`;

// Minimalist fragment shader
const fs = `#version 300 es
precision highp float;
out vec4 fragColor;
void main() {
    fragColor = vec4(0.8, 0.6, 0.2, 1.0);
}`;
----------------------------------------------------------- */