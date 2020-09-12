#include <glad/glad.h>

#include "shader.hpp"
#include "texture.hpp"
#include "vao.hpp"
#include "vbo.hpp"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <stb_image.h>
#include <utility>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

int main() {
  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  // glfw window creation
  // --------------------
  GLFWwindow *window =
      glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "17341039傅禹泽", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // build and compile our shader program
  auto shaderProgram = Shader("assets/vertex.vert", "assets/fragment.frag");

  // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
  std::tuple<glm::vec3, glm::vec3, glm::vec2> vertices[] = {
      std::make_tuple(glm::vec3(-0.5f, -0.5f, 0.0f),
                      glm::vec3(1.0f, 0.0f, 0.0f),
                      glm::vec2(0.0f, 0.0f)), // left
      std::make_tuple(glm::vec3(0.5f, -0.5f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f),
                      glm::vec2(1.0f, 0.0f)), // right
      std::make_tuple(glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f),
                      glm::vec2(0.5f, 1.0f)), // top
  };

  auto vao = VAO();
  auto vbo = VBO();
  // bind the Vertex Array Object first, then bind and set vertex buffer(s), and
  // then configure vertex attributes(s).
  vao.bind();

  vbo.buffer_data(vertices, sizeof(vertices));

  // 位置属性
  // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
  // (void*)0);
  // // 颜色属性
  // glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
  // (void*)(3* sizeof(float)));
  vbo.set_attrib(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), 0);
  vbo.set_attrib(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]),
                 sizeof(glm::vec3));
  vbo.set_attrib(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]),
                 sizeof(glm::vec3) * 2);

  shaderProgram.use();
  auto texture1 = Texture("assets/container.jpg");
  texture1.bind(GL_TEXTURE0);
  auto texture2 = Texture("assets/awesomeface.png");
  texture2.bind(GL_TEXTURE1);
  shaderProgram.setInt("texture1", 0);
  shaderProgram.setInt("texture2", 1);

  // note that this is allowed, the call to glVertexAttribPointer registered VBO
  // as the vertex attribute's bound vertex buffer object so afterwards we can
  // safely unbind
  // glBindBuffer(GL_ARRAY_BUFFER, 0);

  // You can unbind the VAO afterwards so other VAO calls won't accidentally
  // modify this VAO, but this rarely happens. Modifying other VAOs requires a
  // call to glBindVertexArray anyways so we generally don't unbind VAOs (nor
  // VBOs) when it's not directly necessary.
  // glBindVertexArray(0);
  vao.unbind();

  // uncomment this call to draw in wireframe polygons.
  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // render loop
  // -----------
  while (!glfwWindowShouldClose(window)) {
    // input
    // -----
    processInput(window);

    // render
    // ------
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    float timeValue = glfwGetTime();
    float greenValue = sin(timeValue) / 2.0f + 0.5f;

    texture1.bind(GL_TEXTURE0);
    texture2.bind(GL_TEXTURE1);

    // draw our first triangle
    shaderProgram.use();
    shaderProgram.set("ourColor", glm::vec3(0.0f, greenValue, 0.0f));
    vao.bind(); // seeing as we only have a single VAO there's no need to
                // bind it every time, but we'll do so to keep things a
                // bit more organized
    glDrawArrays(GL_TRIANGLES, 0, 3);
    // glBindVertexArray(0); // no need to unbind it every time

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved
    // etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // optional: de-allocate all resources once they've outlived their purpose:
  // ------------------------------------------------------------------------
  // glDeleteVertexArrays(1, &VAO);
  // glDeleteBuffers(1, &VBO);
  // glDeleteProgram(shaderProgram);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
  return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this
// frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  (void)(window);
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}
