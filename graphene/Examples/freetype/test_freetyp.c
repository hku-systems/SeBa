
#include <ft2build.h>
#include FT_FREETYPE_H
#include <sys/time.h>

FT_Library  library;
FT_Face     face;      /* handle to face object */

int freetype_init() {

    auto error = FT_Init_FreeType( &library );

    if ( error )
    {
    //... an error occurred during library initialization ...
        printf("error\n");
    }

    error = FT_New_Face( library,
                    "/usr/share/fonts/truetype/lato/Lato-Light.ttf",
                    0,
                    &face );

    if ( error == FT_Err_Unknown_File_Format )
    {
        //... the font file could be opened and read, but it appears
        //... that its font format is unsupported
        printf("unknown file format\n");
    }
    else if ( error )
    {
        // ... another error code means that the font file could not
        // ... be opened or read, or that it is broken...
        printf("cannot open the file\n");
    }

    error = FT_Set_Char_Size(
            face,    /* handle to face object           */
            0,       /* char_width in 1/64th of points  */
            16*64,   /* char_height in 1/64th of points */
            300,     /* horizontal device resolution    */
            300 );   /* vertical device resolution      */
}

int freetype_check(char* filename)
{
    struct timeval tv;
    fprintf(stderr, "start\n");
    gettimeofday(&tv, 0);
    // freetype_init();
    const char *text = "@abcdefghijklmnopqrstuvwxyz";
    int num_chars = strlen(text);

    FT_Error error = 0;

    FT_GlyphSlot  slot = face->glyph;  /* a small shortcut */
    FT_UInt       glyph_index;
    FT_Bool       use_kerning;
    FT_UInt       previous;
    int           pen_x, pen_y, n;

    pen_x = 300;
    pen_y = 200;

    use_kerning = FT_HAS_KERNING( face );
    previous    = 0;
    // check_words(filename);
    // fprintf(stderr, "ITR %d ", n);

    for (int i = 0;i < 4;i++) {
    for ( n = 0; n < num_chars; n++ )
    {
        /* convert character code to glyph index */
        glyph_index = FT_Get_Char_Index( face, text[n] );

        if (glyph_index < 0) {
            printf("cannot find glyph index\n");
        }

        /* retrieve kerning distance and move pen position */
        if ( use_kerning && previous && glyph_index )
        {
            FT_Vector  delta;


            FT_Get_Kerning( face, previous, glyph_index,
                            FT_KERNING_DEFAULT, &delta );

            pen_x += delta.x >> 6;
        }

        /* load glyph image into the slot (erase previous one) */
        error = FT_Load_Glyph( face, glyph_index, FT_LOAD_RENDER );
        // printf(text[n] << " " << (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec << "\n";
        if ( error ) {
            printf("error during glyph loading, code \n");
            continue;  /* ignore errors */
        }

        /* now draw to our target surface */
        // my_draw_bitmap( &slot->bitmap,
        //                 pen_x + slot->bitmap_left,
        //                 pen_y - slot->bitmap_top );

        /* increment pen position */
        pen_x += slot->advance.x >> 6;

        /* record current glyph index */
        previous = glyph_index;
    }
    }
            gettimeofday(&tv, 0);
        fprintf(stderr, "end\n");
}

int main() {
    freetype_init();
    for (int i = 0;i < 20;i++)
        freetype_check("abc");
}