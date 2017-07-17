import numpy as np

def color_map(map_name):
    if map_name == 'PastelHeat':

        c_map = np.array([115, 85, 176, 116, 87, 177, 117, 89, 179, 118, 91, 180, 120, 92, 181, 121, 94, 182, 122, 96, 184, 123, 98,
             185, 124, 100, 186, 125, 102, 187, 126, 104, 189, 127, 105, 190, 129, 107, 191, 130, 109, 192, 131, 111,
             194, 132, 113, 195, 133, 115, 196, 134, 117, 197, 135, 118, 199, 136, 120, 200, 137, 122, 201, 139, 124,
             202, 140, 126, 204, 141, 128, 205, 142, 130, 206, 143, 131, 207, 144, 133, 209, 145, 135, 210, 146, 137,
             211, 147, 139, 212, 149, 141, 214, 150, 142, 215, 151, 144, 216, 152, 146, 217, 153, 148, 219, 154, 150,
             220, 155, 152, 221, 156, 154, 222, 158, 155, 224, 159, 157, 225, 160, 159, 226, 161, 161, 227, 162, 163,
             229, 163, 165, 230, 164, 167, 231, 165, 168, 232, 166, 170, 234, 168, 172, 235, 169, 174, 236, 170, 176,
             237, 171, 178, 239, 172, 180, 240, 173, 181, 241, 174, 183, 242, 175, 185, 244, 176, 187, 245, 178, 189,
             246, 179, 191, 247, 180, 193, 249, 181, 194, 250, 182, 196, 251, 183, 198, 252, 184, 200, 254, 185, 202,
             255, 186, 203, 255, 187, 204, 255, 188, 205, 255, 189, 205, 255, 190, 206, 255, 191, 207, 255, 192, 207,
             255, 193, 208, 255, 194, 209, 254, 195, 209, 254, 195, 210, 254, 196, 210, 254, 197, 211, 254, 198, 212,
             254, 199, 212, 253, 200, 213, 253, 201, 214, 253, 202, 214, 253, 203, 215, 253, 203, 216, 253, 204, 216,
             252, 205, 217, 252, 206, 218, 252, 207, 218, 252, 208, 219, 252, 209, 219, 251, 210, 220, 251, 211, 221,
             251, 212, 221, 251, 212, 222, 251, 213, 223, 251, 214, 223, 250, 215, 224, 250, 216, 225, 250, 217, 225,
             250, 218, 226, 250, 219, 227, 250, 220, 227, 249, 221, 228, 249, 221, 228, 249, 222, 229, 249, 223, 230,
             249, 224, 230, 249, 225, 231, 248, 226, 232, 248, 227, 232, 248, 228, 233, 248, 229, 234, 248, 230, 234,
             248, 230, 235, 247, 231, 236, 247, 232, 236, 247, 233, 237, 247, 234, 237, 247, 235, 238, 247, 236, 239,
             246, 237, 239, 246, 238, 240, 246, 238, 241, 246, 239, 241, 246, 240, 242, 246, 241, 243, 245, 242, 243,
             245, 243, 244, 245, 244, 244, 244, 244, 244, 243, 244, 243, 241, 244, 243, 240, 244, 243, 239, 244, 242,
             237, 245, 242, 236, 245, 242, 235, 245, 241, 233, 245, 241, 232, 245, 240, 230, 246, 240, 229, 246, 240,
             228, 246, 239, 226, 246, 239, 225, 246, 239, 223, 247, 238, 222, 247, 238, 221, 247, 238, 219, 247, 237,
             218, 247, 237, 216, 248, 237, 215, 248, 236, 214, 248, 236, 212, 248, 236, 211, 248, 235, 209, 249, 235,
             208, 249, 235, 207, 249, 234, 205, 249, 234, 204, 249, 233, 202, 250, 233, 201, 250, 233, 200, 250, 232,
             198, 250, 232, 197, 250, 232, 195, 250, 231, 194, 251, 231, 193, 251, 231, 191, 251, 230, 190, 251, 230,
             189, 251, 230, 187, 252, 229, 186, 252, 229, 184, 252, 229, 183, 252, 228, 182, 252, 228, 180, 253, 227,
             179, 253, 227, 177, 253, 227, 176, 253, 226, 175, 253, 226, 173, 254, 226, 172, 254, 225, 170, 254, 225,
             169, 254, 225, 168, 254, 224, 166, 255, 224, 165, 255, 224, 163, 255, 223, 162, 255, 223, 161, 255, 223,
             159, 255, 222, 158, 255, 222, 156, 255, 220, 155, 255, 217, 153, 254, 214, 151, 254, 212, 150, 253, 209,
             148, 253, 206, 146, 252, 203, 145, 252, 201, 143, 251, 198, 141, 250, 195, 140, 250, 192, 138, 249, 190,
             136, 249, 187, 135, 248, 184, 133, 248, 181, 131, 247, 179, 130, 247, 176, 128, 246, 173, 126, 246, 170,
             125, 245, 168, 123, 244, 165, 121, 244, 162, 119, 243, 159, 118, 243, 157, 116, 242, 154, 114, 242, 151,
             113, 241, 148, 111, 241, 146, 109, 240, 143, 108, 240, 140, 106, 239, 138, 104, 238, 135, 103, 238, 132,
             101, 237, 129, 99, 237, 127, 98, 236, 124, 96, 236, 121, 94, 235, 118, 92, 235, 116, 91, 234, 113, 89, 234,
             110, 87, 233, 107, 86, 232, 105, 84, 232, 102, 82, 231, 99, 81, 231, 96, 79, 230, 94, 77, 230, 91, 76, 229,
             88, 74, 229, 85, 72, 228, 83, 71, 228, 80, 69, 227, 77, 67, 226, 75, 66, 226, 72, 64, 225, 69, 62, 225, 66,
             60, 224, 64, 59, 224, 61, 57, 223, 58, 55, 223, 55, 54, 222, 53, 52, 222, 50, 50, 221, 47, 49])
        return c_map
    if map_name == 'Random':
        c_map = np.asarray([255, 255, 255, 210, 210, 210, 119, 176, 205, 14, 255, 222, 23, 255, 20, 255, 14, 23, 123, 203, 174, 16, 255, 107, 6,
     173, 255, 111, 234, 156, 255, 156, 57, 255, 12, 226, 21, 233, 247, 251, 28, 221, 126, 171, 203, 165, 149, 186, 86,
     138, 255, 123, 237, 140, 60, 194, 246, 36, 255, 101, 171, 112, 217, 184, 137, 179, 255, 2, 127, 195, 142, 163, 20,
     255, 169, 92, 121, 255, 109, 234, 157, 255, 55, 190, 255, 161, 52, 125, 255, 20, 167, 238, 96, 76, 239, 185, 194,
     250, 56, 159, 6, 255, 13, 103, 255, 205, 182, 113, 219, 188, 93, 198, 238, 64, 255, 130, 111, 171, 255, 62, 90,
     178, 232, 196, 115, 189, 100, 255, 140, 159, 161, 180, 233, 176, 91, 223, 21, 255, 255, 132, 61, 195, 136, 169,
     170, 255, 63, 116, 164, 220, 204, 255, 35, 136, 127, 237, 59, 146, 255, 255, 152, 86, 159, 149, 192, 156, 139, 204,
     188, 6, 255, 14, 144, 255, 218, 255, 10, 225, 151, 125, 232, 137, 131, 187, 172, 141, 167, 178, 155, 246, 65, 189,
     113, 166, 222, 190, 121, 190, 140, 211, 149, 198, 105, 197, 207, 255, 5, 223, 93, 184, 66, 239, 196, 126, 156, 218,
     243, 23, 234, 194, 255, 31, 255, 169, 50, 42, 255, 7, 167, 199, 135, 169, 141, 191, 126, 127, 246, 230, 240, 31,
     221, 24, 255, 50, 126, 255, 41, 255, 183, 100, 249, 151, 158, 161, 180, 52, 227, 220, 113, 176, 211, 255, 8, 7, 69,
     255, 109, 255, 196, 38, 255, 106, 118, 21, 255, 207, 216, 152, 132, 255, 157, 27, 255, 20, 219, 204, 19, 255, 88,
     255, 119, 143, 144, 213, 124, 255, 9, 21, 255, 166, 255, 8, 162, 66, 37, 255, 122, 211, 167, 69, 24, 255, 215, 213,
     72, 187, 193, 120, 195, 1, 255, 255, 58, 123, 255, 14, 109, 192, 193, 115, 144, 131, 225, 255, 170, 65, 190, 178,
     132, 138, 10, 255, 75, 255, 65, 127, 129, 245, 19, 255, 20, 255, 93, 62, 136, 178, 186, 210, 192, 98, 181, 82, 236,
     169, 197, 134, 255, 132, 107, 41, 255, 111, 73, 195, 232, 82, 255, 9, 198, 242, 60, 255, 152, 43, 187, 255, 45, 46,
     255, 64, 72, 255, 55, 198, 14, 255, 255, 210, 25, 177, 202, 121, 4, 255, 146, 70, 62, 255, 166, 238, 97, 154, 255,
     50, 255, 217, 16, 4, 255, 216, 255, 207, 16, 255, 29, 199, 133, 182, 185, 55, 247, 198, 149, 29, 255, 111, 74, 255,
     90, 241, 169, 106, 184, 210, 134, 128, 238, 227, 19, 254, 186, 99, 216, 172, 88, 240, 18, 255, 26, 44, 244, 212,
     134, 180, 186, 106, 192, 202, 102, 165, 233, 215, 212, 74, 54, 238, 208, 129, 108, 255, 246, 209, 45, 206, 252, 42,
     234, 66, 199, 225, 206, 69, 228, 159, 113, 131, 197, 172, 160, 230, 110, 127, 153, 220, 116, 255, 9, 42, 255, 201,
     191, 169, 140, 195, 215, 91, 100, 151, 249, 255, 218, 0, 255, 142, 60, 177, 93, 230, 255, 18, 1, 234, 73, 193, 255,
     130, 68, 206, 175, 119, 9, 255, 60, 57, 244, 199, 91, 201, 208, 255, 26, 178, 134, 90, 255, 56, 183, 255, 155, 228,
     118, 247, 216, 37, 73, 237, 190, 168, 137, 194, 171, 90, 238, 21, 255, 107, 153, 210, 136, 143, 209, 147, 245, 210,
     45, 163, 240, 97, 255, 44, 107, 194, 191, 116, 216, 224, 60, 51, 229, 220, 132, 201, 166, 50, 137, 255, 146, 142,
     212, 120, 112, 255, 255, 31, 188, 239, 139, 122, 35, 202, 255, 179, 222, 99, 255, 127, 96, 205, 189, 106, 210, 197,
     94, 105, 255, 111, 169, 145, 185, 224, 183, 93, 186, 255, 30, 10, 255, 164, 63, 244, 193, 229, 169, 102, 198, 232,
     70, 75, 111, 255, 40, 255, 104, 140, 168, 192, 144, 196, 159, 143, 207, 151, 255, 183, 23, 79, 235, 186, 32, 145,
     255, 91, 255, 110, 10, 232, 255, 12, 255, 143, 183, 63, 254, 160, 255, 6, 158, 205, 137, 206, 39, 255, 181, 156,
     163, 219, 112, 169, 87, 133, 255, 255, 36, 206, 205, 171, 123, 255, 3, 160, 196, 143, 161, 255, 158, 75, 137, 156,
     208, 65, 171, 255, 224, 161, 114, 255, 77, 159, 255, 132, 71, 97, 213, 190, 74, 202, 224, 160, 240, 99, 174, 194,
     132, 134, 173, 193, 108, 164, 228, 255, 130, 29, 66, 198, 237, 180, 214, 107])
        return c_map
    else:
        return map_name





