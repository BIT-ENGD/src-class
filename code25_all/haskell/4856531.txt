module Main where

import Graphics.UI.GLUT
import System.Exit (exitWith, ExitCode(ExitSuccess))

reshape :: ReshapeCallback
reshape size = do
   viewport $= (Position 0 0, size)
   matrixMode $= Projection
   loadIdentity
   frustum (-1) 1 (-1) 1 1.5 20
   matrixMode $= Modelview 0

keyboard :: KeyboardMouseCallback
keyboard (Char '\27') Down _ _ = exitWith ExitSuccess
keyboard _            _    _ _ = return ()

renderCube :: Color3 GLfloat -> IO ()
renderCube c = do
   clear [ ColorBuffer ]

   let color3f = color :: Color3 GLfloat -> IO ()
       scalef = scale :: GLfloat -> GLfloat -> GLfloat -> IO ()

   color3f c
   loadIdentity
   lookAt (Vertex3 0 0 5) (Vertex3 0 0 0) (Vector3 0 1 0)
   scalef 1 2 1
   renderObject Wireframe (Cube 1)
   flush

displayR :: DisplayCallback
displayR = renderCube (Color3 1 0 0)

displayB :: DisplayCallback
displayB = renderCube (Color3 0 0 1)

createWindowWithDisplayFunc :: String -> Position -> DisplayCallback -> IO Window
createWindowWithDisplayFunc name pos display = do
   win <- createWindow name
   windowPosition $= pos
   clearColor $= Color4 0 0 0 0
   shadeModel $= Flat
   displayCallback $= display
   reshapeCallback $= Just reshape
   keyboardMouseCallback $= Just keyboard
   return win

main = do
   getArgsAndInitialize
   initialDisplayMode $= [ SingleBuffered, RGBMode ]
   initialWindowSize $= Size 100 100
   initialWindowPosition $= Position 100 100

   createWindowWithDisplayFunc "R" (Position 10 10) displayR
   createWindowWithDisplayFunc "B" (Position 110 10) displayB

   mainLoop

